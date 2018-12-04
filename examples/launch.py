for i in range(20):
    print("tmux new-session -d -s %02d \'./DO.%02d\'" % (i, i))
