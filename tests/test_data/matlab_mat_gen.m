S = dbstack('-completenames');
[pathstr,name,ext] = fileparts(S.file) ;

disp(pathstr);


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
A_t  = A';
AB_t = E';

dlmwrite(pathstr + "/A_1024_32.txt",A(:),'precision',7);
dlmwrite(pathstr +"/B_32_64.txt",B(:),'precision',7);
dlmwrite(pathstr +"/C_1024_32.txt",C(:),'precision',7);
dlmwrite(pathstr +"/AxB_1024_64.txt",E(:),'precision',7);
dlmwrite(pathstr +"/AxC_transposed_1024_1024.txt",D(:),'precision',7);
dlmwrite(pathstr +"/A_transposed_32_1024.txt",A_t(:),'precision',7);
dlmwrite(pathstr +"/AxB_transposed_1024_1024.txt",AB_t(:),'precision',7);
disp("DONE");

