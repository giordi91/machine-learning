A = a + (b-a).*rand([1024,32]);
B = a + (b-a).*rand([32,64]);
C = a + (b-a).*rand([1024,32]);
D = A*(C');
E = A*B;

dlmwrite("/home/giordi/Desktop/A_1024_32.txt",A(:));
dlmwrite("/home/giordi/Desktop/B_32_64.txt",B(:));
dlmwrite("/home/giordi/Desktop/C_1024_32.txt",C(:));
dlmwrite("/home/giordi/Desktop/AxB_1024_1024.txt",E(:));
dlmwrite("/home/giordi/Desktop/AxC_transposed_1024_1024.txt",D(:));
disp("DONE");

