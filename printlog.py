def print(MESSAGE):
f=open("../logs/test.txt", "a")
f.write(MESSAGE)
f.write("\n")
f.close()