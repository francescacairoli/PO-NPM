function [ x ] = rand_state( range, unsafe_reg )
%RAND_STATE Randomly generate a state within a specified range and outside
%a specified unsafe region.
%   range    State bound, a matrix of size n-by-2 whose i-th row specifies
%            the range for x(i).
%   unsafe_region    Unsafe region, a matrix of size n-by-2 whose i-th row
%                    specified the unsafe range for x(i).

n = size(range, 1);
if ~exist('unsafe_reg', 'var') || isempty(unsafe_reg)
    x = zeros(n, 1);
    for i = 1:n
        x(i) = range(i, 1) + (range(i, 2) - range(i, 1)) * rand;
    end
    %x(n+1, 1) = randi([1 2]);
else
    x = unsafe_reg(:, 1);
    while ~issafe(x, unsafe_reg)
        for i = 1:n
            x(i) = range(i, 1) + (range(i, 2) - range(i, 1)) * rand;
        end
    end
    %x(n+1, 1) = randi([1 2]);
end

end

