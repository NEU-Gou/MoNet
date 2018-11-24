function [ now ] = bilinear_simple_forward( layer, pre, now )
%BILINEAR_SIMPLE_FORWARD Summary of this function goes here
%   Detailed explanation goes here

now.x = pre.x'*pre.x;
end

