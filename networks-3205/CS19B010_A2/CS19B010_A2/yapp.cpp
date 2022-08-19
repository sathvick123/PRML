#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <netinet/ip_icmp.h>
#include <time.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include <chrono> 
#include <poll.h>
using namespace std; 



struct ping_packet{
	struct icmphdr hdr;
	char msg[64-sizeof(struct icmphdr)];
};	
  

void send_ping(int sockfd, char * IP){
	struct sockaddr_in distance ; 
	distance.sin_family = AF_INET;
	distance.sin_port = htons(8080);
		
      	if(inet_aton(IP, (struct in_addr *)&distance.sin_addr.s_addr) == 0){
         cout<<"Bad hostname"<<endl;
  	  exit(1);
	}
	
	else
	{
		struct timeval timeout;
		timeout.tv_sec = 4;
		timeout.tv_usec = 0;
		struct ping_packet packet, temp;
		bzero(&packet, sizeof(packet));
		packet.hdr.type = ICMP_ECHO;	
		int set_socket_time = setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char *)&timeout, (int)sizeof(timeout));
		if(set_socket_time<0)// seting the timer
		{
		 cout << "Could not set time"<<endl;
			exit(1);
		}		
		else
		{		  
			auto s= std::chrono::high_resolution_clock::now();// starting the clock
			if(sendto(sockfd, &packet, sizeof(packet), 0, (struct sockaddr *) &distance,sizeof(distance)) < 0){
			  	cout<<"Could not send the packet"<<endl;
				exit(1);
			}
			 
			else
			{
			
				struct pollfd poll_fd;
				poll_fd.fd = sockfd;
				poll_fd.events = POLLIN;
				int response = poll((struct pollfd * ) &poll_fd, 1, 4 * 1000);
				

				if(response==0)
				{
				  	cout<<"Request timed out or host unreacheable"<<endl;
					exit(1);
				}
				else
				{
					int addr_len = sizeof(temp);
					recvfrom(sockfd, &packet, sizeof(packet), 0, (struct sockaddr*)&temp, (socklen_t*)&addr_len);	// receieve the packet
					auto e = std::chrono::high_resolution_clock::now();			// stoping the clock to do the calculation of time difference
					auto roundtriptime = std::chrono::duration<double, std::milli>(e-s);
					cout<<"Response from "<< IP <<".";
					cout<<"Round Trip Time = " << roundtriptime.count() << " ms"<<endl;                         				
				}
			}			 			
		}		 	
     }	
}

int main(int argc, char *argv[]){

	int sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_ICMP);

	if (sockfd != -1) {
            char *IP=argv[1];
	send_ping(sockfd,IP);		
	}
      else{
	printf("socket creation failed...\n");
		exit(0);
		}
	return 0;
}
