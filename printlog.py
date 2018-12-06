def print(MESSAGE):
	f=open("logs/test.txt", "a+")
	f.write(str(MESSAGE))
	f.write("\n")
	f.close()