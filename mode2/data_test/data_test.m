clc;clear;
%% 单点异常测试
x=[1,2,3,4,5,6,7,8,9,10];
y=linspace(500,400,10);
y(5)=0; % 在中间插入一个0值，模拟突发情况

data=[x;y];
% plot(x,y);

% for i=1:3
%     filter_with_sigma(y, i)
% end
for i=1:3
    filter_with_MAD(y, i)
    filter_with_MAD_window(y, i, 10)
end


%% 直线测试
disp("---------直线测试---------");
y=linspace(500,0,1000);
% y=y+randn(size(y))*10;
% for i=1:3
%     filter_with_sigma(y, i)
% end
for i=1:3
    filter_with_MAD(y, i)
    filter_with_MAD_window(y, i, 10)
end
plot(y)

%% 多次充值测试
disp("---------多次充值测试---------");
y=[linspace(300,20,200),linspace(400,2,1000),linspace(300,1,1000)];
y=y+randn(size(y))*5;
y(randi([1,length(y)],10,1))=0;
find(y==0)
% for i=1:3
%     filter_with_sigma(y, i)
% end
[upper_bound, lower_bound]=filter_with_MAD_window(y, 3, 10);
plot(y)
hold on;
plot(upper_bound, 'r--');
plot(lower_bound, 'g--');



%% 真实数据测试
% 实际需要考虑数据点很少的情况，所以需要严格把控数据范围？当数据点小于10的时候，不进行异常值过滤。
disp("---------真实数据测试---------");
y=[482,480,470,465,460,455,450,445,440,435,430,425,420,415,410,405,400,395,390,385,380,375,370,365,360,355,350,345,340,335,330,325,320,315,310,305,300,295,290,285,280,275,270,265,260,255,250,245,240,235,230,225,220,215,210,205,200,195,190,185,180,175,170,165,160,155,150];
y=y+randn(size(y))*1;
y([5,10])=0;
y=[y,[500:-4.6:490]];
find(y==0)
[upper_bound, lower_bound]=filter_with_MAD_window(y, 3, 10);
plot(y)
hold on;
plot(upper_bound, 'r--');
plot(lower_bound, 'g--');




function filter_with_sigma(data, sigma_level)
    fprintf("使用mu±%dsigma进行过滤...\n",sigma_level)
    mu=mean(data);
    sigma=std(data);
    upper_bound=mu+sigma_level*sigma;
    lower_bound=mu-sigma_level*sigma;
    filtered_data=data(data>=lower_bound & data<=upper_bound);
    fprintf("上界：%f\n", upper_bound);
    fprintf("下界：%f\n", lower_bound);
    % fprintf("检测到异常值：%d\n", length(data)-length(filtered_data));
    fprintf("过滤后数据点数量: %d / %d\n", length(filtered_data), length(data));
    fprintf("\n");
end




function filter_with_MAD(data, threshold)
    fprintf("使用MAD进行过滤，阈值：%f...\n", threshold)
    med=median(data);
    MAD=median(abs(data-med));
    upper_bound=med+threshold*MAD;
    lower_bound=med-threshold*MAD;
    filtered_data=data(data>=lower_bound & data<=upper_bound);
    fprintf("上界：%f\n", upper_bound);
    fprintf("下界：%f\n", lower_bound);
    % fprintf("检测到异常值：%d\n", length(data)-length(filtered_data));
    fprintf("过滤后数据点数量: %d / %d\n", length(filtered_data), length(data));
    fprintf("\n");
end

% 同时返回上下界
function [upper_bound_list, lower_bound_list]=filter_with_MAD_window(data, threshold, window_size)
    fprintf("使用滑动窗口MAD进行过滤，阈值：%f，窗口大小：%d...\n", threshold, window_size)
    filtered_data = data; % 初始化过滤后的数据
    upper_bound_list = zeros(1,length(data));
    lower_bound_list = zeros(1,length(data));
    half_window = floor(window_size / 2);
    
    for i = 1:length(data)
        % 定义窗口范围
        start_idx = max(1, i - half_window);
        end_idx = min(length(data), i + half_window);
        window_data = data(start_idx:end_idx);
        
        med = median(window_data);
        MAD = median(abs(window_data - med));
        upper_bound = med + threshold * MAD;
        lower_bound = med - threshold * MAD;
        upper_bound_list(i) = upper_bound;
        lower_bound_list(i) = lower_bound;
        
        if data(i) < lower_bound || data(i) > upper_bound
            % 突发充值，保留
            if i > 1 && data(i) > data(i-1)
                continue;
            end
            filtered_data(i) = NaN; % 标记为异常值
        end
    end
    
    % 移除NaN值
    fprintf("检测到异常值：%d\n")
    disp(find(isnan(filtered_data)))
    filtered_data = filtered_data(~isnan(filtered_data));
    fprintf("过滤后数据点数量: %d / %d\n", length(filtered_data), length(data));
    fprintf("\n");
end