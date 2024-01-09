<div align="center">

# Image Generating Subnet <!-- omit in toc -->

---

</div>

## Introduction
Welcome to the Image Generating Subnet project. This README provides an overview of the project's structure and example usage for both validators and miners.

### The Incentivized Internet
- [Discord](https://discord.gg/bittensor)
- [Network](https://taostats.io/)
- [Research](https://bittensor.com/whitepaper)

## Project Structure
- `image_generation_subnet`: Contains base, feature functions, and utilities for validators and miners.
- `neurons`: Contains the validator and miner loop.
- `dependency_modules`: Includes servers for `prompt_generation`, `rewarding`, and `miner_endpoint`.

## Installation
1. Clone the repository.
```bash
git clone https://github.com/NicheTensor/NicheImage.git
```
2. Install the dependencies.
```bash
cd NicheImage
pip install -r requirements.txt
```
3. Install the project.
```bash
pip install -e .
```

## Example Usage
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

## Start Miner
Before running the following commands, make sure to replace the placeholder arguments with appropriate values.

First you need to start an image generation API on a gpu server that your miners can use. A RTX 3090 GPU is enough for several miners.
```bash
python dependency_modules/miner_endpoint/app.py --port <port> --model_name <model_name>
```

You can also run with pm2. For example like this for SDXLTurbo:
```bash
pm2 start python --name "image_generation_endpoint_SDXLTurbo" -- -m dependency_modules.miner_endpoint.app --port 10006 --model_name SDXLTurbo
```

Or, you can start the RealisticVision model like this:
```bash
pm2 start python --name "image_generation_endpoint_RealisticVision" -- -m dependency_modules.miner_endpoint.app --port 10006 --model_name RealisticVision
```

Then you can run several miners using the image generation API:
```bash
pm2 start python --name "miner" \
-- \
-m neurons.miner.miner \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--generate_endpoint <your_miner_endpoint>/generate \
--info_endpoint <your_miner_endpoint>/info \
--axon.port <your_public_port> \
```

You can also start with pm2, here is an example:
```bash
pm2 start python --name "miner" -- -m neurons.miner.miner --netuid 23 --wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> --subtensor.network finney --generate_endpoint http://127.0.0.1:10006/generate --info_endpoint http://127.0.0.1:10006/info --axon.port 10010
```

**View logs** 
```bash
pm2 logs miner
```

## Start Validator

Requirements: A validator only needs a cpu server to validate by using our free to use APIs for checking image generation. This is the default setting and requires no configuration.

However, it is possible to run your own image checking APIs if you prefer. This does require a GPU with min 20 GB of ram. You can see how to do this [here.](./dependency_modules/README.md)

If validators opt in to share their request capacity they will get paid for each image they generate. Opt in is done by specifying --proxy.port
If passed, a proxy server will start that allows us to query through their validator, and the validator will get paid weekly for the images they provide.

### Start Validator with Default Settings

```bash
pm2 start python --name "validator_nicheimage" \
-- -m neurons.validator.validator \
--netuid <netuid> \
--wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> \
--subtensor.network <network> \
--axon.port <your_public_port> \
--proxy.port <other_public_port> # Optional, pass only if you want allow queries through your validator and get paid
```

**View logs** 
```bash
pm2 logs validator_nicheimage
```

### Schedule update and restart validator
Pull the latest code from github and restart the validator every hour.
```bash
pm2 start auto_update.sh --name "auto-update" --cron-restart="0 * * * *" --attach
```

# Roadmap

We will release updates on Tuesdays, in order to make it predictable for when changes to the network will be introduced. Furhter we will do our best to share updates in advance.

Here is the current roadmap for the subnet:

9th of January: Launch of stable version of repo, simple demo frontend released.

16th of January: Release paid API for buying images from the network, profit directly distributed to validators. Additional model introduced to the network. 

23rd of January: Launch improved frontend.

February: Add categories where miners can compete with any model of their choice, and incentive is calculated based on how well their images compares to other miners. We will add categories such as "realistic photographs". This will allow the network to always have the best available models in each category, and incentivize people to create even better image generation models.
