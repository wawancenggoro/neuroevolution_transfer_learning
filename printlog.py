import datetime

def printlog(MESSAGE):

	currentDT = datetime.datetime.now()
	filename = str(currentDT.day)+"-"+str(currentDT.month)+"-"+str(currentDT.year)+".txt"

	print(MESSAGE)
	f=open("logs/"+filename, "a+")
	f.write(str(MESSAGE))
	f.write("\n")
	f.close()