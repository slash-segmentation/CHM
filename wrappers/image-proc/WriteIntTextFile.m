function WriteIntTextFile(A, file)
if ischar(file); id = fopen(file,'w'); else; id = file; end;
fprintf(id,[repmat('%d ',1,size(A,2)) '\n'],A);
if ischar(file); fclose(id); end;
