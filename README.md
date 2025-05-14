# Simple Diffusion Model in Rust

A minimal, single-file implementation of a diffusion model in Rust using pure Rust crates. This project demonstrates the forward and reverse diffusion processes, a simplified U-Net denoiser, and a basic training loop for CIFAR-10 (placeholders).

## Features

- Linear beta scheduler for diffusion
- Simplified U-Net denoiser with timestep conditioning (placeholder)
- Forward and reverse diffusion processes (DDPM sampling)
- Training loop for CIFAR-10 dataset (placeholder)
- Command-line interface (CLI) for training or sampling

## Requirements

- Rust (nightly toolchain recommended)
- Cargo (Rust's package manager)

## Dependencies

The project uses the following Rust crates:

- [`ndarray`](https://crates.io/crates/ndarray) for numerical computations
- [`rand`](https://crates.io/crates/rand) and [`rand_distr`](https://crates.io/crates/rand_distr) for random number generation
- [`image`](https://crates.io/crates/image) for image processing
- [`clap`](https://crates.io/crates/clap) for CLI argument parsing

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/sadhusoumik/SimpleDiffusionMod.git
   cd rust-diffusion-model
   ```
