from subprocess import check_call

names = open("pyro-contributors.txt").read().strip().split()
for name in names:
    print(name)
    check_call(["yarn", "all-contributors", "add", name, "code"])
