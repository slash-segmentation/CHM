function A=ReadIntTextFile(file)
A=dlmread(file);
A=A(:,1);
