
xrange_orig = [-11, 11];
yrange_orig = [-23, 13];
xrange = [-23, 23];
yrange = xrange;

curr_dir = [0, -1];
lim_x = [-1, 1];
lim_y = [-1, 1];

x = 0;
y = 0;
visited = [x, y];
idx = 0;
while true
  if x >= xrange_orig(1) && x <= xrange_orig(2) && y >= yrange_orig(1) && y <= yrange_orig(2)
    idx = idx + 1;
    visited(idx, :) = [x, y];
    disp([x, y, curr_dir, lim_x, lim_y, xrange, yrange])
  %   pause
  else
    disp(['skip', x, y])
  end
  
  if curr_dir(1) == 0 && curr_dir(2) == -1
%     Going up
    y = y-1;
    if y <= lim_y(1)
      y = lim_y(1);
      lim_y(1) = max(lim_y(1) - 1, yrange(1)-1);
      curr_dir = [-1, 0];
    end
  else
    if curr_dir(1) == -1 && curr_dir(2) == 0
      % Going left
      x = x-1;
      if x <= lim_x(1)
        x = lim_x(1);
        lim_x(1) = max(lim_x(1) - 1, xrange(1)-1);
        curr_dir = [0, 1];
      end
    else
      if curr_dir(1) == 0 && curr_dir(2) == 1
        %Going down
        y = y + 1;
        if y >= lim_y(2)
          y = lim_y(2);
          lim_y(2) = min(lim_y(2) + 1, yrange(2)+1);
          curr_dir = [1, 0];
        end
      else
        if curr_dir(1) == 1 && curr_dir(2) == 0
          % Go right
          x = x+1;
          if x >= lim_x(2)
            x = lim_x(2);
            lim_x(2) = min(lim_x(2) + 1, xrange(2)+1);
            curr_dir = [0, -1];
          end
        end
      end
    end
  end
  if (x <= xrange(1) || x >= xrange(2)) && (y <= yrange(1) || y >= yrange(2))
    disp('Abort')
    disp([x, y, xrange, yrange])
    break;
  end
end

hold off
plot(visited(:,1), visited(:,2), '-')
hold on
plot(visited(:,1), visited(:,2), 'x')
