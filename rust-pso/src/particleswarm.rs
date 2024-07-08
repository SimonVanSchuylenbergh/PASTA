// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Particle Swarm Optimization (PSO)
//!
//! Canonical implementation of the particle swarm optimization method as outlined in \[0\] in
//! chapter II, section A.
//!
//! For details see [`ParticleSwarm`].
//!
//! ## References
//!
//! \[0\] Zambrano-Bigiarini, M. et.al. (2013): Standard Particle Swarm Optimisation 2011 at
//! CEC-2013: A baseline for future PSO improvements. 2013 IEEE Congress on Evolutionary
//! Computation. <https://doi.org/10.1109/CEC.2013.6557848>
//!
//! \[1\] <https://en.wikipedia.org/wiki/Particle_swarm_optimization>

use crate::fitting::SCALING;
use crate::interpolate::Bounds;
use argmin::{
    argmin_error, argmin_error_closure,
    core::{ArgminFloat, CostFunction, Error, PopulationState, Problem, Solver, SyncAlias, KV},
    float,
};
use argmin_math::{ArgminRandom, ArgminSub};

use nalgebra as na;
use rand::{Rng, SeedableRng};

#[derive(Clone)]
pub struct ParticleSwarm<B: Bounds, F, R> {
    /// Inertia weight
    weight_inertia: F,
    /// Cognitive acceleration coefficient
    weight_cognitive: F,
    /// Social acceleration coefficient
    weight_social: F,
    /// Delta (potential)
    delta: F,
    /// Bounds on parameter space
    bounds: B,
    /// Number of particles
    num_particles: usize,
    /// Random number generator
    rng_generator: R,
}

impl<B, F> ParticleSwarm<B, F, rand::rngs::StdRng>
where
    B: Bounds,
    F: ArgminFloat,
{
    pub fn new(bounds: B, num_particles: usize) -> Self {
        ParticleSwarm {
            weight_inertia: float!(1.0f64 / (2.0 * 2.0f64.ln())),
            weight_cognitive: float!(0.5 + 2.0f64.ln()),
            weight_social: float!(0.5 + 2.0f64.ln()),
            delta: float!(1e-7),
            bounds,
            num_particles,
            rng_generator: rand::rngs::StdRng::from_entropy(),
        }
    }
}
impl<B, F, R0> ParticleSwarm<B, F, R0>
where
    B: Bounds,
    F: ArgminFloat,
    R0: Rng,
{
    pub fn with_rng_generator<R1: Rng>(self, generator: R1) -> ParticleSwarm<B, F, R1> {
        ParticleSwarm {
            weight_inertia: self.weight_inertia,
            weight_cognitive: self.weight_cognitive,
            weight_social: self.weight_social,
            delta: self.delta,
            bounds: self.bounds,
            num_particles: self.num_particles,
            rng_generator: generator,
        }
    }
}

const N: usize = 5;
type P = na::SVector<f64, 5>;

impl<B, F, R> ParticleSwarm<B, F, R>
where
    B: Bounds,
    F: ArgminFloat,
    R: Rng,
{
    pub fn with_inertia_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: inertia factor must be >=0."
            ));
        }
        self.weight_inertia = factor;
        Ok(self)
    }

    pub fn with_cognitive_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: cognitive factor must be >=0."
            ));
        }
        self.weight_cognitive = factor;
        Ok(self)
    }

    pub fn with_social_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: social factor must be >=0."
            ));
        }
        self.weight_social = factor;
        Ok(self)
    }

    pub fn with_delta(mut self, delta: F) -> Result<Self, Error> {
        if delta < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: delta must be >=0."
            ));
        }
        self.delta = delta;
        Ok(self)
    }

    /// Initializes all particles randomly and sorts them by their cost function values
    fn initialize_particles<O: CostFunction<Param = P, Output = F> + SyncAlias>(
        &mut self,
        problem: &mut Problem<O>,
    ) -> Result<Vec<Particle<P, F>>, Error> {
        let (positions, velocities) = self.initialize_positions_and_velocities();

        let costs = problem.bulk_cost(&positions)?;

        let mut particles = positions
            .into_iter()
            .zip(velocities)
            .zip(costs)
            .map(|((p, v), c)| Particle::new(p, c, v))
            .collect::<Vec<_>>();

        // sort them, such that the first one is the best one
        particles.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(particles)
    }

    /// Initializes positions and velocities for all particles
    fn initialize_positions_and_velocities(&mut self) -> (Vec<P>, Vec<P>) {
        let (min, max) = self.bounds.minmax();
        let delta = max.sub(&min).component_div(&SCALING);
        let delta_neg = -delta;
        let positions = self
            .bounds
            .generate_random_within_bounds(&mut self.rng_generator, self.num_particles)
            .into_iter()
            .map(|p| p.component_div(&SCALING))
            .collect();
        let velocities = (0..self.num_particles)
            .map(|_| P::rand_from_range(&delta_neg, &delta, &mut self.rng_generator))
            .collect();
        (positions, velocities)
    }
}

fn random_uniform(rng: &mut impl Rng, a: f64, b: f64) -> f64 {
    if a == b {
        a
    } else if a < b {
        rng.gen_range(a..b)
    } else {
        rng.gen_range(b..a)
    }
}

fn calculate_potential<const N: usize>(
    particle: &Particle<na::SVector<f64, N>, f64>,
    global_best: f64,
    dim: usize,
) -> f64 {
    particle.velocity[dim].abs() + (global_best - particle.position[dim]).abs()
}

impl<B, O, R> Solver<O, PopulationState<Particle<P, f64>, f64>> for ParticleSwarm<B, f64, R>
where
    B: Bounds,
    O: CostFunction<Param = P, Output = f64> + SyncAlias,
    f64: ArgminFloat,
    R: Rng,
{
    const NAME: &'static str = "ParticleSwarm";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<P, f64>, f64>,
    ) -> Result<(PopulationState<Particle<P, f64>, f64>, Option<KV>), Error> {
        // Users can provide a population or it will be randomly created.
        let particles = match state.take_population() {
            Some(mut particles) if particles.len() == self.num_particles => {
                // sort them first
                particles.sort_by(|a, b| {
                    a.cost
                        .partial_cmp(&b.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                particles
            }
            Some(particles) => {
                return Err(argmin_error!(
                    InvalidParameter,
                    format!(
                        "`ParticleSwarm`: Provided list of particles is of length {}, expected {}",
                        particles.len(),
                        self.num_particles
                    )
                ))
            }
            None => self.initialize_particles(problem)?,
        };

        Ok((
            state
                .individual(particles[0].clone())
                .cost(particles[0].cost)
                .population(particles),
            None,
        ))
    }

    /// Perform one iteration of algorithm
    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<P, f64>, f64>,
    ) -> Result<(PopulationState<Particle<P, f64>, f64>, Option<KV>), Error> {
        let mut best_particle = state.take_individual().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No current best individual in state."
        ))?;
        let mut best_cost = state.get_cost();
        let mut particles = state.take_population().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No population in state."
        ))?;

        for p in 0..particles.len() {
            let mut new_position = na::SVector::<f64, 5>::zeros();
            for d in 0..N {
                let condition = particles
                    .iter()
                    .all(|p| calculate_potential(p, best_particle.position[d], d) < self.delta);
                if condition {
                    // forced update
                    // println!("Forced update in d={}, i={}, p={}", d, state.iter, p);
                    particles[p].velocity[d] =
                        random_uniform(&mut self.rng_generator, -self.delta, self.delta);
                } else {
                    // Regular PSO update
                    let momentum = particles[p].velocity[d] * self.weight_inertia;

                    let to_optimum = particles[p].best_position[d] - particles[p].position[d];
                    let pull_to_optimum = random_uniform(&mut self.rng_generator, 0.0, to_optimum)
                        * self.weight_cognitive;

                    let to_global_optimum = best_particle.position[d] - particles[p].position[d];
                    let pull_to_global_optimum =
                        random_uniform(&mut self.rng_generator, 0.0, to_global_optimum)
                            * self.weight_social;

                    particles[p].velocity[d] = momentum + pull_to_optimum + pull_to_global_optimum;
                    new_position[d] = particles[p].position[d] + particles[p].velocity[d];
                }
            }
            particles[p].position = self
                .bounds
                .clamp(new_position.component_mul(&SCALING))
                .component_div(&SCALING);
        }
        let positions: Vec<P> = particles.iter().map(|p| p.position).collect();

        let costs = problem.bulk_cost(&positions)?;

        for (p, c) in particles.iter_mut().zip(costs.into_iter()) {
            p.cost = c;

            if p.cost < p.best_cost {
                p.best_position = p.position;
                p.best_cost = p.cost;

                if p.cost < best_cost {
                    best_particle.position = p.position;
                    best_particle.best_position = p.position;
                    best_particle.cost = p.cost;
                    best_particle.best_cost = p.cost;
                    best_cost = p.cost;
                }
            }
        }

        Ok((
            state
                .individual(best_particle)
                .cost(best_cost)
                .population(particles),
            None,
        ))
    }

    fn terminate(
        &mut self,
        state: &PopulationState<Particle<P, f64>, f64>,
    ) -> argmin::core::TerminationStatus {
        let particles = state.get_population().unwrap();
        let global_best = match state.best_individual.as_ref() {
            Some(p) => p.position,
            None => return argmin::core::TerminationStatus::NotTerminated,
        };
        let condition = particles.iter().all(|p| {
            for d in 0..N {
                if p.position[d] - global_best[d] > self.delta {
                    return false;
                }
            }
            true
        });
        if condition {
            argmin::core::TerminationStatus::Terminated(
                argmin::core::TerminationReason::SolverConverged,
            )
        } else {
            argmin::core::TerminationStatus::NotTerminated
        }
    }
}

/// A single particle
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Particle<T, F> {
    /// Position of particle
    pub position: T,
    /// Velocity of particle
    velocity: T,
    /// Cost of particle
    pub cost: F,
    /// Best position of particle so far
    best_position: T,
    /// Best cost of particle so far
    best_cost: F,
}

impl<T, F> Particle<T, F>
where
    T: Clone,
    F: ArgminFloat,
{
    pub fn new(position: T, cost: F, velocity: T) -> Particle<T, F> {
        Particle {
            position: position.clone(),
            velocity,
            cost,
            best_position: position,
            best_cost: cost,
        }
    }
}
