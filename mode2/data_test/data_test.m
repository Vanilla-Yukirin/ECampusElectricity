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
    filter_with_MAD_window_buffer(y, i, 10, 50, 3)
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
    filter_with_MAD_window_buffer(y, i, 10, 50, 3)
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
filter_with_MAD_window_buffer(y, 3, 10, 80, 4);
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


%% 更现实的数据测试
disp("---------真实数据测试2---------");
data=generate_test_data(500, 4.6, 0);
data=[data, generate_test_data(500, 4.6, 490)];
data=[data, generate_test_data(490, 4.6, 0)];
data=[data, generate_test_data(500, 4.6, 490)];

[upper_bound, lower_bound]=filter_with_MAD_window(data, 3, 10);
figure(5);
plot(data)
hold on;
plot(upper_bound, 'r--');
plot(lower_bound, 'g--');
[upper_bound, lower_bound]=filter_with_MAD_window_buffer(data, 3, 10, 500, 5);
figure(6);
plot(data)
hold on;
plot(upper_bound, 'r--');
plot(lower_bound, 'g--');

% 生成模拟数据，输入是 数据数据起点，预估每一步下降多少元，数据终点。虽然长度可能不固定，但是可以预估长度
function data=generate_test_data(st, step, ed)
    data=[];
    current=st;
    while current>ed
        data=[data, current];
        current=current-(step+randn()*0.5);
        disp(data)
    end
end


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

% 同时返回上下界和过滤后的数据
% 在 mode2/data_test/data_test.m 的三段测试循环里新增 filter_with_MAD_window_buffer 调用，直接把“突增缓冲”策略跑进现有样本，参数默认 threshold=3、window=10、increase_buffer≈50~80、buffer_points=3~4，可以直观看到充值场景下的过滤效果。
function [upper_bound_list, lower_bound_list, filtered_data]=filter_with_MAD_window_buffer(data, threshold, window_size, increase_buffer, buffer_points)
    fprintf("使用MAD+缓冲区进行过滤，阈值：%f，窗口大小：%d，缓冲区增量：%f，缓冲区点数：%d...\n", threshold, window_size, increase_buffer, buffer_points)
    filtered_data = data;
    upper_bound_list = zeros(1,length(data));
    lower_bound_list = zeros(1,length(data));
    half_window = floor(window_size / 2);
    buffer_countdown = 0;
    
    for i = 1:length(data)
        start_idx = max(1, i - half_window);
        end_idx = min(length(data), i + half_window);
        window_data = data(start_idx:end_idx);
        
        med = median(window_data);
        MAD = median(abs(window_data - med));
        upper_bound = med + threshold * MAD;
        lower_bound = med - threshold * MAD;
        upper_bound_list(i) = upper_bound;
        lower_bound_list(i) = lower_bound;
        
        current_value = data(i);
        
        if current_value < lower_bound % 异常低值，直接标记删除
            filtered_data(i) = NaN;
            continue;
        end
        
        if current_value > upper_bound % 异常高值，判断是否为突增充值
            % 如果比前一个数高，且小于缓冲区阈值，则判定为突增充值
            is_recharge = i > 1 && current_value > data(i-1) && (current_value - upper_bound) <= increase_buffer;
            if is_recharge || buffer_countdown > 0 % 突增充值，或者最近充值（缓冲区内）
                buffer_countdown = buffer_points;
                continue;
            else
                filtered_data(i) = NaN;
                disp(i)
                continue;
            end
        end
        
        if buffer_countdown > 0
            buffer_countdown = buffer_countdown - 1;
        end
    end
    
    fprintf("检测到异常值：%d\n", length(find(isnan(filtered_data))))
    filtered_data = filtered_data(~isnan(filtered_data));
    fprintf("过滤后数据点数量: %d / %d\n", length(filtered_data), length(data));
    fprintf("\n");
end
