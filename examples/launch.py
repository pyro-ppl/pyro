for i in range(10):
    print("tmux new-session -d -s %02d \'./DO.%02d\'" % (i, i))
