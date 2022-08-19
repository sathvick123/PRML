
#include<bits/stdc++.h>
#include <iostream>
#include<stdio.h>
#include <string.h> //strlen
#include <stdlib.h>
#include <errno.h>
#include <unistd.h> //close
#include <arpa/inet.h> //close
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros
#include <chrono>
#include <ctime>
#include <time.h>
#include <fcntl.h>
#include <signal.h>
#include <netinet/ip_icmp.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#define MAX 80
#define PORT 8080
#define SA struct sockaddr
using namespace std;
char buff[MAX];

void read_from_server(int sockfd)
{
	  read(sockfd,buff,sizeof(buff));// 1 read
	  if(buff[0]=='_'||buff[0]=='X'||buff[0]=='O')
	  {
	     for(int i=0;i<3;i++)
	   {
	    for(int j=0;j<3;j++)
	    {
	     printf("%c",buff[i*3+j]);
	     if(j<2)
	     {
	      printf("|");
	     }
	    
	    }
	    printf("\n");
	   }
	   printf("\n");
	  }
	
	  
}

int write_from_client(int client,int sockfd)
{         
        int i,j;
        printf("Enter (ROW, COL) for placing your mark: ");
 	
 	//int t1,t2;
 	///clock fn ...........................................
 	
 	auto end = std::chrono::high_resolution_clock::now();
 	auto start = std::chrono::high_resolution_clock::now();
 	auto t = std::chrono::duration<double, std::milli>(end-start);
 	scanf("%d%d",&i,&j);
 	
 	// 
 	if((t.count())/1000>15)
 	{
 	  strcpy(buff,"exceeded");
 	  write(sockfd,buff,sizeof(buff));
 	  return -2;
 	}
 	else
 	{
          strcpy(buff,"notexceeded");
 	  write(sockfd,buff,sizeof(buff)); 	
 	}
        bzero(buff, sizeof(buff));  
 	buff[0]=i+'0';
  	buff[2]=j+'0';
  	  write(sockfd,buff,sizeof(buff));//1 write
	  bzero(buff, sizeof(buff));  
	  return 0;
}

void printt(char ch)
{
   	  if(ch=='b')
          {
            printf("Draw match\n");
          }
          if(ch=='c')
          {
            printf("Player 1 won the match\n ");
          }
           if(ch=='d')
          {
           printf("Player 2 won the match\n ");
          }
}

void printmat()
{
for(int i=0;i<3;i++)
	  {
	   for(int j=0;j<3;j++)
	   {
	     printf("_");
	     if(j<2)
	     {
	      printf("|");
	     }    
	   }
	   printf("\n");
	  }

}
/*****************************************************************************************************************/
void start(int c,int sockfd,int x)
{
  int i,j;
  char buff[MAX];
  if(c==1)///
  {
        
	if(x==1)
	{
	  printf("Your player ID is 1.and your symbol is 'X' . .\n");
	} 
	
	 
      for(;;)
     {
        int x=write_from_client(1,sockfd);//write from client 1 to server
        if(x==-2)
        {
          return;
        }
        read_from_server(sockfd);//read and print the data from the server
        read(sockfd,buff,sizeof(buff));//state 
        
        if(buff[0]>'a')
        {
          printt(buff[0]);
          return;
        }
	read_from_server(sockfd);//know that write has happend from client2	
	read(sockfd,buff,sizeof(buff));//state
	if(buff[0]>'a')
        {
          printt(buff[0]);
          return;
        }
	
     }
	   
  }
  else  if(c==2) ////
  {
       for(;;)
        {
          read_from_server(sockfd);//to know that client 1 read the input
          read(sockfd,buff,sizeof(buff));//state
	  if(buff[0]>'a')
	  {
      	   printt(buff[0]);
	   return;
	  }
          
          int x=write_from_client(2,sockfd);// take input in client 2 and write to server
		  if(x==-2)
		{
		  return;
		}
          read_from_server(sockfd);//print the updated matrix from server
          read(sockfd,buff,sizeof(buff));//state 
	   if(buff[0]>'a')
	   {
      	    printt(buff[0]);
	    return;
	   }
      }
  }
  
}
/*****************************************************************************************************************/
int main()
{
	int sockfd, connfd, c;
	struct sockaddr_in servaddr, cli;
       
	// socket create and verification
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd == -1) {
		printf("socket creation failed...\n");
		exit(0);
	}
	bzero(&servaddr, sizeof(servaddr));

	// assign IP, PORT
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	servaddr.sin_port = htons(PORT);

	// connect the client socket to server socket
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) {
		printf("connection with the server failed...\n");
		exit(0);
	}
	
	read(sockfd,buff,sizeof(buff));//client 1 ready
	int cl=buff[0]-'0';
	
	int x=0;
	if(cl==1)
	{
	   
	   printf("Connected to the game server. Your player ID is 1.\nWaiting for a partner to join . . .\n");
	   read(sockfd,buff,sizeof(buff));//
	   printf("Your partner's ID is 2. Your symbol is ‘X’.\nStarting the game …\n");
	   printmat();
	}
	else
	{
	  printf("Connected to the game server. Your player ID is 2.\nYour partner’s ID is 1. Your symbol is ‘O’\nStarting the game …\n");
          printmat();
	}
	   
	  for(;;)
	  {
	   start(cl,sockfd,x);
	   x=1;
	   printf("Do you want to continue: ");
	  scanf("%d",&c);
          
          bzero(buff, sizeof(buff));  
          buff[0]=c+'0';
          //printf("buff[0]: %c\n",buff[0]);
          write(sockfd,buff,sizeof(buff));//status given player
           bzero(buff, sizeof(buff));  
           read(sockfd,buff,sizeof(buff));//overall status
           char p=buff[0];	
           if(p=='n')
           {
            exit(0);
           }
	   
	    
	  }
	
	
	
	
	// close the socket
	close(sockfd);
}

