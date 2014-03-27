function x=ParseArgument(x)
% Parses a string argument as either a scalar, logical, vector, or matrix. This
% is very % useful for functions that are compiled since their arguments are
% always passed as strings. This is performed in a safe manner as to not
% execute arbitrary code (like str2num does natively). If the input is not a
% string or is not parsable, it is returned as-is.
if ischar(x)
  xx = ((x >= '0') & (x <= '9')) | ismember(x,'.+-');
  if all(xx); x=str2double(x); % scalar, safest conversion
  elif strcmp(lower(x),'true')  || strcmp(lower(x),'t'); x = true;  % logical true, also very safe
  elif strcmp(lower(x),'false') || strcmp(lower(x),'f'); x = false; % logical false, also very safe
  elseif x(1) == '[' && x(end) == ']'
    % vector or matrix, strip brackets and check for delimiters 
    xx = xx(2:end-1) | ismember(x(2:end-1),',; '); % no parentheses or letters!
    if all(xx); x=str2num(x(2:end-1)); end; % we have made str2num safe with all our checks
  end
end
