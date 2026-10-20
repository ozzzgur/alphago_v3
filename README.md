File Responsibilities
game_env.py : This acts as the strict referee for the game of Go. All other scripts must use it to check the rules and legal moves.
train.py : Using the neural network from model.py, it processes 5,000 human games with the help of data_processor.py. This step creates our first working brain, saved as alphago_model_v3.pth.
self_play_data.py : The bot uses alphago_model_v3.pth to play against itself and explore 200 moves deep. This process generates entirely new training data from its own experience.
self_train.py : It takes the trained model weights and mixes them with the new self-play data. This combination creates our final, much smarter model named hybrid.pth.
last_agent.py: This script connects our trained model directly to the internet. It opens the cosumi.net website to play live matches against the bot there.
