function out = Mymaxpooling(in,L)

if L==0, out = in; return; end

if mod(size(in,1),2)~=0
    in = [in; zeros(1,size(in,2),size(in,3))];
end;
if mod(size(in,2),2)~=0
    in = [in zeros(size(in,1),1,size(in,3))];
end;
in = max(max(in(1:2:end,1:2:end,:),in(1:2:end,2:2:end,:)),max(in(2:2:end,1:2:end,:),in(2:2:end,2:2:end,:)));
out = Mymaxpooling(in,L-1);
