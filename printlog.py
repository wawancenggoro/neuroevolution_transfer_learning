def print(MESSAGE):
	f=open("logs/test.txt", "a+")
	f.write(MESSAGE)
	f.write("\n")
	f.close()

def print(MESSAGE,MESSAGE_ARG):
	f=open("logs/test.txt", "a+")
	f.write(MESSAGE)
	f.write("\n")
	f.write(str(MESSAGE_ARG))
	f.write("\n")
	f.close()