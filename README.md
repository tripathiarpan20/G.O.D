<h1 align="center">G.O.D Subnet</h1>


🚀 Welcome to the [Gradients on Demand](https://finetuning-web.vercel.app/app) Subnet

> Providing access to Bittensor network for on-demand training at scale.


## Setup Guides

- [Miner Setup Guide](docs/miner_setup.md)
- [Validator Setup Guide](docs/validator_setup.md)

## Recommended Compute Requirements

### Validator Requirements

A validator should be capable of running inference on multiple 8B models in parallel.

| Component    | Specification                          |
|-------------|---------------------------------------|
| VRAM        | 80GB                                  |
| Storage     | 500GB (minimum)<br>1TB (recommended)  |
| RAM         | 64GB+ (recommended)                   |
| Example GPUs| A100 (recommended)<br>H100 (overkill) |
| vCPUs       | 12 (minimum)<br>18+ (recommended)     |

### Miner Requirements

Requirements vary based on accepted jobs and LoRA training usage.

| Component    | Specification                           |
|-------------|----------------------------------------|
| VRAM        | 80+GB                                   |
| Storage     | 500GB (minimum)<br>1TB (recommended)   |
| RAM         | 64GB+ (recommended)                    |
| Example GPUs| A100 (minimal)<br>4xH100 (recommended) |
| vCPUs       | 12 (minimum)<br>18+ (recommended)      |
