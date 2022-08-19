function FnDrawMcsHeatMap(UeMCS,varargin)
% Function FnDrawMcsHeatMap, draws a heat map of MCS values derived based
% on te SINR values at that position.
% UeMCS: The MCS distribution for given 2D region.

%%
% The minimum work - i.e. ploting.

fig = figure();
axis tight
hold on;

nVarargs = length(varargin);

if (nVarargs > 0)
    tit = varargin{1};
    xdim = varargin{2};
    ydim = varargin{3};
%     xlimval = varargin{4};
%     ylimval = varargin{5};
    xtick = varargin{4}; % #### 6};
    xticklabels = varargin{5}; % #### 7};
    set(fig, 'Position', [100 100 xdim ydim]);
end

surf(UeMCS,'EdgeColor','none','LineStyle','none','FaceLighting','phong');

view([0,90]);

% Number of unique MCS values.
% Assumption the nUnq will indicate unique MCS values between 15 and (15 -
% (nUnq) + 1)
nUnq = size(unique(UeMCS),1);

if (nUnq == 16)
    colorbar('Ticks',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'Outage','MCS 1','MCS 2','MCS 3','MCS 4', 'MCS 5', ...
        'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 15)
    
    colorbar('Ticks',[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 1','MCS 2','MCS 3','MCS 4', 'MCS 5', ...
        'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 14)
    
    colorbar('Ticks',[3,4,5,6,7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 2','MCS 3','MCS 4', 'MCS 5', ...
        'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
    
elseif (nUnq == 13)
    
    colorbar('Ticks',[4,5,6,7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 3','MCS 4', 'MCS 5', ...
        'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 12)
    
    colorbar('Ticks',[5,6,7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 4', 'MCS 5', ...
        'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 11)
    
    colorbar('Ticks',[6,7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 5', ...
        'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 10)
    
    colorbar('Ticks',[7,8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 6','MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 9)
    
    colorbar('Ticks',[8,9,10,11,12,13,14,15,16],...
        'TickLabels',{'MCS 7','MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
elseif (nUnq == 8)
    
    colorbar('Ticks',[1,2,3,4,5,6,7,8],...
        'TickLabels',{'MCS 8','MCS 9','MCS 10', 'MCS 11', ...
        'MCS 12','MCS 13','MCS 14','MCS 15'});
    
end


%%
% Additional work based on the varargs

%fprintf('Inputs in varargin(%d):\n',nVarargs)
if (nVarargs > 0) % Assume for now either args are given or not, if given all are given
    
    %title(tit);
%     xlim(xlimval);
%     ylim(ylimval);
    ax = gca;
    % xtick = [1, 188,376];
    % xticklabels = {'-1500','0', '1500'};
    
    set(ax, 'XTick', xtick);
    set(ax, 'XTickLabel', xticklabels);
    set(ax, 'YTick', xtick);
    set(ax, 'YTickLabel', xticklabels);
    
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times','FontSize',10,'FontWeight','bold')
    
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','FontSize',10,'FontWeight','bold')
end
end

