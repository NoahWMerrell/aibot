## Overview

**Project Title**: AIBot

**Project Description**: A simple chatbot that can converse with the user, akin to ChatGPT.

**Project Goals**: To provide a chatbot with a complex enough model that can respond to the user's input in a somewhat intelligible way.

## Instructions for Build and Use

Steps to build and/or run the software:

1. Install Python (at least version 3.13) on their [website](https://www.python.org/).
2. Ensure pip is installed on your computer, following the instructions provided at [this site](https://pip.pypa.io/en/stable/installation/).
2. Once finished, run a few commands in the terminal to ensure you have the appropriate libraries. Note, commands in the terminal may vary depending on the operating system. An easy way to identify if there's differences is to ask for a similar command for your operating system in an LLM (such as [ChatGPT](https://chatgpt.com/)). First, make sure to run ```python -m pip install --upgrade pip setuptools wheel``` in the terminal.
3. Next, run: ```python -m pip install transformers```
4. After that, run: ```python -m pip install torch```
5. Finally, run: ```python -m pip install huggingface_hub[hf_xet]```
6. Once complete, you should have the necessary libraries to run each AIBot

Instructions for using the software:

1. Run the model of your choice, **except model 3** which does not work and was provided to simply show an attempt on getting it working.
2. It will begin by downloading items the model depends on. Do not interrupt this process. These are stored in your computer's cache. On Windows computers this is typically located at ```C:\Users\yourusername\.cache\huggingface\```.
2. You will be prompted to enter whatever you wish.
3. The model will respond appropriately (results vary widely as models 1 & 2 and fairly lightweight).

## Development Environment 

To recreate the development environment, you need the following software and/or libraries with the specified versions:

* Python 3.13
* Transformers
* Torch
* huggingface_hub[hf_xet] (Helps things run smoothly)

## Useful Websites to Learn More

I found these websites useful in developing this software:

* [Python](https://www.python.org/)
* [Pip Installation](https://pip.pypa.io/en/stable/installation/)
* [ChatGPT](https://chatgpt.com/) (Was especially helpful for getting things started with this assignment)

## Future Work

The following items I plan to fix, improve, and/or add to this project in the future:

* [ ] Allow for the input of additional training data
* [ ] Utilize a more advanaced model to perform more complex tasks (completing model 3)
* [ ] Improve efficiency