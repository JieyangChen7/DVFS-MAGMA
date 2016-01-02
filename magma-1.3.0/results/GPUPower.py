import sys, time

def main():
	if len(sys.argv) <= 1:
		print ("Wrong command!")
		print ("Usage: phyton <program_name.py> <file_name.txt>")
		print ("""Or phyton <program_name.py> -h for help information""")
		sys.exit(0)

	if "-h" in sys.argv[1]:
		print("""This program will input 1 file and output 2 files.
		\r\nIn GPUPower.py, there are 3 methods:\n1. main()\n2. printOutput1(4 parameters)\n3. printOutput2(4 parameters)
		\r\nIn main(), input file will be read and processed.\nThen, printOutput1() will output first file.
		\rThe file will list time in linux time, and power of the two attached GPUs
		\r\nLastly, printOutput2() will output second file.
		\rThe second file will print total time, and total power of the two GPUs""")
		sys.exit(0)
		
	fi = open(sys.argv[1], 'r')
	#print "filename: ", fi.name
	fo1 = open("gpupower1.out", 'w')
	fo2 = open("gpupower2.out", 'w')

	timeStamp = []
	nameList = []	
	power = []

	for line in fi:
		lineList = []

		if line.startswith("Timestamp"):
			line = line.replace("Timestamp                       : ", "")
			##timeStamp.append(line.strip())
			#print timeStamp
	
		elif line.startswith("GPU"):
			lineList = line.split(":")
			name = lineList[1]
			
			if name not in nameList:
				nameList.append(name)
				#print name
		
		elif "Power Draw" in line:
			lineList = line.split(":")
			lineList[1] = lineList[1].replace(" W","")
			power.append(float(lineList[1].strip()))
			#print power
	fi.close()
	##printOutput1(fo1, timeStamp, nameList, power)			
	printOutput2(fo2, timeStamp, nameList, power)
	#print "closed or not: ", fi.closed

def printOutput1(fo1, timeStamp, nameList, power):
	fo1.write("Timestamp(s)")

	for i in range(len(nameList)):
		fo1.write(" GPU"+nameList[i]+ "(W)")

	j = 0
	for i in range(len(timeStamp)):
		timeStamp[i] = time.mktime(time.strptime(timeStamp[i], "%a %b %d %H:%M:%S %Y"))
		fo1.write("\n%10.2f" %timeStamp[i])
		
		for k in range(len(nameList)):
			fo1.write(" %3.2f" %power[j])
			j += 1
			
	fo1.close()

def printOutput2(fo2, timeStamp, nameList, power):
	fo2.write("Time Duration(s)\tTotal Energy(J)\n\t\t\t\t ")

	for i in range(len(nameList)):
		fo2.write("   GPU"+nameList[i])
		
	totEne1 = 0.0
	totEne2 = 0.0
	totTime = 0.0
	prevTime = 0.0
	j = 0	
	for i in range(len(timeStamp)):
		prevTime = timeStamp[i] - 1.0
		totTime += (timeStamp[i] - prevTime)
		totEne1 += ((timeStamp[i] - prevTime)*power[j])
		totEne2 += ((timeStamp[i] - prevTime)*power[j+1])
		#print "%f"%totEne2
		j += len(nameList)
		prevTime = timeStamp[i]
		
	fo2.write("\n%2.2f" %(totTime-1.0))
	fo2.write("\t\t\t   %4.2f" %totEne1)
	fo2.write("   %4.2f" %totEne2) 
	fo2.close()
	
main()
