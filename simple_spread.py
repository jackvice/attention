import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box


from flax import struct

# 1. Update the State dataclass
@struct.dataclass
class State:
    p_pos: chex.Array
    p_vel: chex.Array
    c: chex.Array
    done: chex.Array
    step: int
    landmark_drift_vel: chex.Array            # [L, 2]
    landmark_cover_streak: chex.Array = struct.field(default=None)
    # NEW: Add previous landmark positions
    landmark_prev_pos: chex.Array = struct.field(default=None)  # [L, 2]


# 2. Update observation space in __init__
class SimpleSpreadMPE(SimpleMPE):
    def __init__(
            self,
            num_agents=3,
            num_landmarks=3,
            local_ratio=0.5,
            action_type=DISCRETE_ACT,
            agent_radius: float = 0.05,
            landmark_radius: float = 0.05,
            **kwargs,
    ):
        dim_c = 2  # NOTE follows code rather than docs

        # Action and observation spaces
        agents = ["agent_{}".format(i) for i in range(num_agents)]
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        # MODIFIED: Add extra num_landmarks*2 for previous positions
        observation_spaces = {
            i:Box(-jnp.inf, jnp.inf, (4+(num_agents-1)*4+(num_landmarks*4),)) 
            for i in agents
        }

        colour = [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks

        # Env specific parameters
        self.local_ratio = local_ratio
        assert (
            self.local_ratio >= 0.0 and self.local_ratio <= 1.0
        ), "local_ratio must be between 0.0 and 1.0"

        # Parameters
        rad = jnp.concatenate(
            #[jnp.full((num_agents), 0.15), jnp.full((num_landmarks), 0.05)] # jmv
            [jnp.full((num_agents), agent_radius), jnp.full((num_landmarks), landmark_radius)]
        )
        collide = jnp.concatenate(
            [jnp.full((num_agents), True), jnp.full((num_landmarks), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=dim_c,
            colour=colour,
            rad=rad,
            collide=collide,
            **kwargs,
        )


    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Initialise with random positions and landmark drift velocities"""
    
        key_a, key_l, key_drift = jax.random.split(key, 3)

        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0),
        ])

        # Initialize random drift directions with magnitude 2.5
        drift_angles = jax.random.uniform(key_drift, (self.num_landmarks,), minval=0,
                                          maxval=2*jnp.pi)
        landmark_drift_vel = 0.3 * jnp.stack([jnp.cos(drift_angles), jnp.sin(drift_angles)],
                                             axis=1)
        cover_streak0 = jnp.zeros((self.num_landmarks,), dtype=jnp.int32)
    
        # NEW: Initialize previous landmark positions to current positions
        landmark_positions = p_pos[self.num_agents:]
        landmark_prev_pos = landmark_positions.copy()

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            landmark_drift_vel=landmark_drift_vel,
            landmark_cover_streak=cover_streak0,
            landmark_prev_pos=landmark_prev_pos,  # NEW
        )

        return self.get_obs(state), state
        
    
    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # Update landmark positions with drift before physics step
        state = self._update_landmark_drift(state)
        
        # Call parent step_env with updated state
        return super().step_env(key, state, actions)


    def _update_landmark_drift(self, state: State) -> State:
        """Update landmark positions based on drift, stopping for coverage or boundaries"""

        # NEW: Save current landmark positions as previous positions
        current_landmark_pos = state.p_pos[self.num_agents:]
        
        # Check which landmarks are covered by agents using collision detection radius
        def _is_covered(landmark_idx: int) -> bool:
            landmark_pos = state.p_pos[self.num_agents + landmark_idx]
            agent_distances = jnp.linalg.norm(state.p_pos[:self.num_agents] -
                                              landmark_pos, axis=1)
            # Use collision detection radius as requested
            coverage_threshold = self.rad[:self.num_agents] + self.rad[self.num_agents +
                                                                       landmark_idx]
            return jnp.any(agent_distances <= coverage_threshold)
        
        covered = jax.vmap(_is_covered)(jnp.arange(self.num_landmarks))
        # covered: [L] already computed above

        # update streak (consecutive coverage count)
        new_streak = jnp.where(covered, state.landmark_cover_streak + 1, 0).astype(jnp.int32)

        landmark_pos = state.p_pos[self.num_agents:]
        drift_step = state.landmark_drift_vel * self.dt * ~covered[:, None]
        new_landmark_pos = landmark_pos + drift_step

        boundary_buffer = 0.08
        at_boundary = jnp.abs(new_landmark_pos) >= (1.75 - boundary_buffer)
        clamped_pos = jnp.clip(new_landmark_pos, -1.75 + boundary_buffer, 1.75 - boundary_buffer)

        # freeze permanently if (streak>=3) OR (hit boundary on any axis)
        #freeze_now = jnp.logical_or(new_streak >= 3, jnp.any(at_boundary, axis=1))  # [L]
        freeze_now = jnp.any(at_boundary, axis=1)  # [L]
        new_drift_vel = jnp.where(freeze_now[:, None], 0.0, state.landmark_drift_vel)  # [L,2]

        new_p_pos = jnp.concatenate([state.p_pos[:self.num_agents], clamped_pos])
        return state.replace(
            p_pos=new_p_pos,
            landmark_drift_vel=new_drift_vel,
            landmark_cover_streak=new_streak,  # persist streak
            landmark_prev_pos=current_landmark_pos,  
        )


        
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx: int):
            """Values needed in all observations"""

            landmark_pos = (
                state.p_pos[self.num_agents :] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame


            # NEW: Previous landmark positions in agent reference frame
            landmark_prev_pos = (
                state.landmark_prev_pos - state.p_pos[aidx]
            )  # Previous landmark positions in agent reference frame
            
            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]
            comm = jnp.roll(
                state.c[: self.num_agents], shift=self.num_agents - aidx - 1, axis=0
            )[: self.num_agents - 1]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            comm = jnp.roll(comm, shift=aidx, axis=0)

            return landmark_pos, landmark_prev_pos, other_pos, comm

        landmark_pos, landmark_prev_pos, other_pos, comm = _common_stats(self.agent_range)

        def _obs(aidx: int):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    landmark_prev_pos[aidx].flatten(),  # NEW: num_landmarks * 2
                    other_pos[aidx].flatten(),  # 5, 2
                    comm[aidx].flatten(),
                ]
            )

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    
    def rewards(self, state: State) -> Dict[str, float]:
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(
            self.agent_range,
            self.agent_range,
        )  # [agent, agent, collison]

        def _agent_rew(aidx: int, collisions: chex.Array):
            rew = -1 * jnp.sum(collisions[aidx])
            return rew

        def _land(land_pos: chex.Array):
            d = state.p_pos[: self.num_agents] - land_pos
            return -1 * jnp.min(jnp.linalg.norm(d, axis=1))

        global_rew = jnp.sum(jax.vmap(_land)(state.p_pos[self.num_agents :]))

        rew = {
            a: _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio)
            for i, a in enumerate(self.agents)
        }
        return rew
