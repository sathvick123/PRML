
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


 ofstream fptr("fn.log");



// Function designed for chat between client and server.

 char buff[MAX];
 char matrix[3][3];
void init()
{
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<3;j++)
    {
     matrix[i][j]='_';
    }
  }
  
} 
 
 
int read_from_client(int client,int connfd1,int connfd2) 
{
   int i,j;
   if(client==1)
   {      
      read(connfd1,buff,sizeof(buff));
      int x=strcmp("exceeded",buff);
      if(x==0)
      {
          return -2;
      }
      read(connfd1,buff,sizeof(buff));//read from c1
      
       fptr<<"Player1 made this move:"<<buff[0]<<" "<<buff[2]<<endl<<endl<<"matrix: "<<endl;
       for(int i=0;i<3;i++)
       {
        for(int j=0;j<3;j++)
        {
          fptr<<matrix[i][j];
          if(j<2)
          {
          fptr<<"|";
          }
        }
        fptr<<endl;
       }
       fptr<<endl;
         i=buff[0]-'0';
         j=buff[2]-'0';
         matrix[i-1][j-1]='X';//update   
         return 0;      
   }
   else
   {
       read(connfd2,buff,sizeof(buff));
      int x=strcmp("exceeded",buff);
      if(x==0)
      {
          return -2;
      }
       read(connfd2,buff,sizeof(buff));//read from c2
       fptr<<"Player2 made this move:"<<buff[0]<<" "<<buff[2]<<endl<<endl;
       
       for(int i=0;i<3;i++)
       {
        for(int j=0;j<3;j++)
        {
          fptr<<matrix[i][j];
          if(j<2)
          {
          fptr<<"|";
          }
        }
        fptr<<endl;
       }
      
         i=buff[0]-'0';
         j=buff[2]-'0';
         matrix[i-1][j-1]='O';//update
         return 0;
   }

}
  
int state()
{
  for(int i=0;i<3;i++)
  {
    if((matrix[i][0]==matrix[i][1])&&(matrix[i][0]==matrix[i][2]))
    {
      if(matrix[i][0]=='X')
      {
       return 1;
      }
      if(matrix[i][0]=='O')
      {
        return 2;
      }
    }
  }
  
    for(int i=0;i<3;i++)
  {
    if((matrix[0][i]==matrix[1][i])&&(matrix[0][i]==matrix[2][i]))
    {
      if(matrix[0][i]=='X')
      {
       return 1;
      }
      if(matrix[0][i]=='O')
      {
        return 2;
      }
    }
  }
  
  if(matrix[0][0]==matrix[1][1]&&matrix[1][1]==matrix[2][2])
  {
    if(matrix[0][0]=='X')
      {
       return 1;
      }
     if(matrix[0][0]=='O')
      {
        return 2;
      }
  }
  
  if(matrix[2][0]==matrix[1][1]&&matrix[1][1]==matrix[0][2])
  {
    if(matrix[1][1]=='X')
      {
       return 1;
      }
      if(matrix[1][1]=='O')
      {
        return 2;
      }
  }
  
  for(int i=0;i<3;i++)
  {
   for(int j=0;j<3;j++)
   {
     if(matrix[i][j]!='O'&&matrix[i][j]!='X')
     {
       return -1;
     }
   }
  }
  return 0;
  
} 
 

 /******************************************************/
int play(int connfd1,int connfd2)
{     
       for(;;)
       {
          bzero(buff,MAX);
          int x=read_from_client(1,connfd1,connfd2);//read from client1
          if(x==-2)
          {
           return -2;
          }
          bzero(buff,MAX);
          for(int i=0;i<3;i++)
          {    for(int j=0;j<3;j++)
               { 
                 buff[i*3+j]=matrix[i][j];     
               }
          }
         
          write(connfd1,buff,sizeof(buff)); //write to client1
	  write(connfd2,buff,sizeof(buff)); //write to client2
          bzero(buff,MAX);
          
          int st=state();
          buff[0]='a'+st+1;
          write(connfd1,buff,sizeof(buff)); //state
          write(connfd2,buff,sizeof(buff)); //state
          if(st!=-1)
          {
            return st;
          }   
          
          x=read_from_client(2,connfd1,connfd2); //read from client2
          if(x==-2)
          {
           return -2;
          }
          
           
          bzero(buff,MAX);
           for(int i=0;i<3;i++)
          {    for(int j=0;j<3;j++)
               { 
                 buff[i*3+j]=matrix[i][j];     
               }
          }
          write(connfd1,buff,sizeof(buff)); //write to client 1
	   write(connfd2,buff,sizeof(buff));//write to client 2
           
           st=state();
           
             buff[0]='a'+st+1;
          write(connfd1,buff,sizeof(buff)); //state
          write(connfd2,buff,sizeof(buff)); //state 
          if(st!=-1)
          {
           return st;
          }
                           
       }  
  
}
/***************************************************************************************/
// Driver function
int main()
{
       init();
       
      

       
       
	int sockfd;
	unsigned int len1,len2;
     int connfd1,connfd2,cli;
	struct sockaddr_in servaddr, cli1,cli2;//

	// socket create and verification
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd == -1) {
		printf("socket creation failed...\n");
		exit(0);
	}
	
	bzero(&servaddr, sizeof(servaddr));

	// assign IP, PORT
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servaddr.sin_port = htons(PORT);

	// Binding newly created socket to given IP and verification
	if ((bind(sockfd, (SA*)&servaddr, sizeof(servaddr))) != 0) {
		printf("socket bind failed...\n");
		exit(0);
	}
	
	// Now server is ready to listen and verification
	if ((listen(sockfd, 5)) != 0) {
		printf("Listen failed...\n");
		exit(0);
	}
	else
		printf("Game server started. Waiting for players.\n");
	len1 = sizeof(cli1);
        len2 = sizeof(cli2);
        
       // Accept the data packet from client and verification
	connfd1 = accept(sockfd, (SA*)&cli1, &len1);
	
	if (connfd1 < 0) {
		printf("server accept failed...1\n");
		exit(0);
	}
	bzero(buff, MAX);
	buff[0]='1';
	write(connfd1,buff,1);/////////////////////////////////
	
     connfd2 =accept(sockfd, (SA*)&cli2, &len2);	
     if (connfd2 < 0) {
		printf("server accept failed...2\n");
		exit(0);
	}
	 bzero(buff, MAX);
	buff[0]='2';
	cli=1;
        write(connfd1,buff,sizeof(buff));//to know that client 2 is ready
        auto start = std::chrono::high_resolution_clock::now(); 
        
        
       fptr<<"Game has started  "<<endl<<endl;
       write(connfd2,buff,sizeof(buff));/////////////////////
        for(;;)
        {
          char c,d;
          int ress=play(connfd1,connfd2);
          
          auto end = std::chrono::high_resolution_clock::now();			// stop the clock to calculate time difference
	  auto tt = std::chrono::duration<double, std::milli>(end-start);
					
          
          fptr<<"Game ended  "<<endl<<endl;/////////////////////////////////////////////////////////////////////////////////////////doubt
          
          fptr<<"Game lasted for a time of : "<<tt.count()/1000<<endl<<endl;
          if(ress!=-2)
          {
          	  if(ress==0)
		 {
		  fptr<<"Game is draw"<<endl<<endl;
		 }
		 if(ress>0)
		 {
		   fptr<<"Game won by player "<<ress<<endl<<endl;
		 }
          }
         
          read(connfd1,buff,sizeof(buff));//client1
          c=buff[0];
          bzero(buff, sizeof(buff));  
          read(connfd2,buff,sizeof(buff));//client 2
          d=buff[0];
           bzero(buff, sizeof(buff)); 
          if(c=='0'||d=='0')
          {
           buff[0]='n';
            write(connfd1,buff,sizeof(buff));
            write(connfd2,buff,sizeof(buff));
            
              auto end = std::chrono::high_resolution_clock::now();
              auto tim = std::chrono::duration<double, std::milli>(end-start); 
              int cou=tim.count();
              fptr<<"Game time ="<<cou/1000<<endl<<endl;
              
            exit(0);
          }
          else
          {
             start = std::chrono::high_resolution_clock::now();
             fptr<<"New Game has started : "<<endl<<endl;//////////////////////////////////
             
            buff[0]='y';
            write(connfd1,buff,sizeof(buff));
            write(connfd2,buff,sizeof(buff));
          }
          
          init();
          
        }        
        
	close(sockfd);
}
