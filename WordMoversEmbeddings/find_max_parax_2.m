function [max_value_for_parax] = find_max_parax_2(parax, steps, sample_num)
    max_value_for_parax = zeros(1,sample_num);
    counter_for_max_parax = 1;
    for i=1:steps:length(parax)
        max_value_for_parax(counter_for_max_parax) = max(parax(1,i:i+steps-1));
        counter_for_max_parax = counter_for_max_parax+1;
    end
end