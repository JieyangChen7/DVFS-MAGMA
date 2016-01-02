#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


int main(int argc, char* argv[]){
	if(argc != 4){
		printf("usage: filename case# write id\n");
		return -1;
	}
	FILE* total;
	FILE* cpu;
	FILE* gpu;
	
	size_t x = 0;
	char* bf = (char*) malloc(sizeof(char) * 512);
	char* t = (char*) malloc(sizeof(char) * 512);
	char* c = (char*) malloc(sizeof(char) * 512);
	//char* g = (char*) malloc(sizeof(char) * 512);
	
	strncpy(t, argv[1], strlen(argv[1]));
	strncpy(c, argv[1], strlen(argv[1]));
	//strcpy(g, "python GPUPower.py ");
	//strncpy(g + strlen("python GPUPower.py "), argv[1], strlen(argv[1]));
	strcpy(t + strlen(argv[1]), ".total.pwr");
	strcpy(c + strlen(argv[1]), ".cpu.pwr");
	//strcpy(g + strlen(argv[1]) + strlen("python GPUPower.py "), ".gpu.pwr");
	int read;

	total = fopen(t, "r");
	cpu = fopen(c, "r");

	double totalen = 0;
	while(getline(&bf, &x, total) != -1){
		int i = 0, space = 0;
		while(space < 2){
			if(isspace(bf[i]))
				space++;
			i++;
		}
		totalen += atof(&bf[i]);
	}

	double pkg1 = 0, pkg2 = 0;
	getline(&bf, &x, cpu);
	while(getline(&bf, &x, cpu) != -1){
		int i = 0;
		while(isspace(bf[i])) i++;
		while(!isspace(bf[i])) i++;
		while(isspace(bf[i])) i++;
		while(!isspace(bf[i])) i++;
		while(isspace(bf[i])) i++;
		pkg1 += atof(&bf[i]);

		while(!isspace(bf[i])) i++;
		while(isspace(bf[i])) i++;
                while(!isspace(bf[i])) i++;
		while(isspace(bf[i])) i++;
                while(!isspace(bf[i])) i++;
		while(isspace(bf[i])) i++;
                while(!isspace(bf[i])) i++;
		while(isspace(bf[i])) i++;
		pkg2 += atof(&bf[i]);
	}
	
	double gpu1 = 0, gpu2 = 0, gpu3 = 0, gpu4 = 0;
	/*int check = system(g);
	if(check < 0){
		printf("system call failed, no gpu output\n");
		return -1;
	}*/

	gpu = fopen("gpupower1.out", "r");
	//getline(&bf, &x, gpu);
	while(getline(&bf, &x, gpu) != -1){
		int i = 0;
		//while(!isspace(bf[i])) i++;
                //while(isspace(bf[i])) i++;
		gpu1 += 0.1*atof(&bf[i]);
		/*while(!isspace(bf[i])) i++;
                while(isspace(bf[i])) i++;
		gpu2 += atof(&bf[i]);
		while(!isspace(bf[i])) i++;
                while(isspace(bf[i])) i++;
                gpu3 += atof(&bf[i]);
                while(!isspace(bf[i])) i++;
                while(isspace(bf[i])) i++;
                gpu4 += atof(&bf[i]);*/
	}	

	fclose(total); fclose(cpu); fclose(gpu);
	free(bf); free(t); free(c); //free(g);
	system("rm *.out");
	int idn = atoi(argv[3]);

	printf("total energy: %f\nCPU1: %f\nCPU2: %f\ntotal CPU: %f\nGPU1: %f\nGPU2: %f\nGPU3: %f\nGPU4: %f\ntotal GPU: %f\n",
		 totalen, pkg1, pkg2, pkg1+pkg2, gpu1, gpu2, gpu3, gpu4,  gpu1+gpu2+gpu3+gpu4);
	char title[128];
	if(idn > 0) sprintf(title,"results%i.txt",idn);
	else sprintf(title,"results.txt");
	if(atoi(argv[2]) == 1){
		FILE* res = fopen(title, "r+");
		if(res == NULL){
			printf("couldn't write results\n");
			return -1;
	}
		fseek(res, 0, SEEK_END);
		fprintf(res, "%g\t\t%g\t\t%g\t\t%g\t\t\t%g\t\t%g\t\t%g\t\t%g\t\t%g\n", totalen, pkg1, pkg2, pkg1+pkg2, gpu1, gpu2, gpu3, gpu4, gpu1+gpu2+gpu3+gpu4);
		fclose(res);
	}
	return 0;


}
