a = -1;
b = 1;
A = single(a + (b-a).*rand([1024,32]));
B = single(a + (b-a).*rand([32,64]));
C = single(a + (b-a).*rand([1024,32]));
D = (A*(C'))';
E = (A*B)';


A=A';
B=B';
C=C';

disp(A(1,1));
dlmwrite("/home/giordi/Desktop/A_1024_32.txt",A(:),'precision',7);
dlmwrite("/home/giordi/Desktop/B_32_64.txt",B(:),'precision',7);
dlmwrite("/home/giordi/Desktop/C_1024_32.txt",C(:),'precision',7);
dlmwrite("/home/giordi/Desktop/AxB_1024_1024.txt",E(:),'precision',7);
dlmwrite("/home/giordi/Desktop/AxC_transposed_1024_1024.txt",D(:),'precision',7);


disp("DONE");

