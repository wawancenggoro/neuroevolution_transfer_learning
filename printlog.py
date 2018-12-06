import datetime

def print(MESSAGE):

	currentDT = datetime.datetime.now()
	filename = str(currentDT.day+"-"+currentDT.month+"-"+currentDT.year+".txt")

	print(MESSAGE)
	f=open("logs/"+filename, "a+")
	f.write(str(MESSAGE))
	f.write("\n")
	f.close()