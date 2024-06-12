Group 3 - author: Adnan Al Saad

This is related to an application for playing the Stackelberg game for COMP34612.
This file explains how to use the leaders files in the game.

First of all, there are dependencies you must install. All of them are installable via pip:
    * scikit-learn
    * pandas
    * statsmodels
    * numpy

You can install it using the instruction pip install name. where name is substituted with the elements after *

Our group uses the multiple leaders approach and are used as follows
- leader1.py is used for mk1
- leader2.py is used for mk2
- leader3.py is used for mk3

After running the main.py file and have the app window opened, execute the leader file using python file-name. For example, python leader1.py. And then press on the connect button, then a green text saying 'connected' will appear and the program is ready to be used. The leader file should be on the same file of the app because base_leader is inherited by the leader files.