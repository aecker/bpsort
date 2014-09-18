classdef Layout
    properties
        name    % name of the probe
        x       % x coordinates of the channels (top is zero, positive values go down)
        y       % y coordinates
        n       % number of electrodes
    end
    
    methods
        function self = Layout(in)
            % Create Layout object.
            %   layout = Layout(file) creates a Layout object from the
            %   given config file. The n^th row of the config file contains
            %   the x and y coordinates for the n^th channel, separated by
            %   a tab.
            %
            %   layout = Layout(coords) creates a Layout object using the
            %   given matrix of coordinates. The first column contains the
            %   x coordinates, the second column the y coordinates. The
            %   n^th row contains the coordinates for the n^th channel.
            
            assert(isnumeric(in) || ischar(in), 'Input must be either a config file name or an N-by-2 matrix!')
            if ischar(in)
                file = in;
                fid = fopen(file);
                if fid
                    data = fscanf(fid, '%g\t%g\n');
                    self.name = file;
                    self.x = data(1 : 2 : end);
                    self.y = data(2 : 2 : end);
                    fclose(fid);
                else
                    error('Could not open config file %s!', file)
                end
            else
                assert(size(in, 2) == 2, 'Numeric input must be N-by-2 matrix!')
                xy = in;
                self.x = xy(:, 1);
                self.y = xy(:, 2);
            end
            self.n = numel(self.x);
        end
        
        function order = channelOrder(self, coord)
            % Sort channels
            %   order = layout.channelOrder('x') returns the channel
            %   indices ordered by their x coordinate (analogous for y).
            %
            %   order = layout.channelOrder('yx') orders first by y, then
            %   by x.
            
            assert(all(ismember(coord, 'xy')), 'Input must specify combination of x and y!')
            m = max(max(self.x), max(self.y));
            k = 0;
            nc = numel(coord);
            for i = 1 : nc
                k = k + m ^ (nc - i) * eval(['self.' coord(i)]);
            end
            [~, order] = sort(k);
        end
        
        function k = neighbors(self, i, d)
            % Return list of neighbors
            %   k = neighbors(self, i, d) returns a list of neighbors for
            %   channel i, defined as those channels with a distance less
            %   than d (excluding channel i itself).
            
            x = self.x;
            y = self.y;
            k = find((x - x(i)) .^ 2 + (y - y(i)) .^ 2 < d ^ 2);
            k = setdiff(k, i);
        end
        
        function plot(self, hdl)
            % Plot layout.
            %   layout.plot() plots the layout of the probe.
            %
            %   layout.plot(handle) uses the given figure or axes handle to
            %   produce the plot.
            
            if nargin < 2
                hdl = 0;
            end
            switch get(hdl, 'type')
                case 'figure'
                    figure(hdl)
                case 'axes'
                    axes(hdl)
                otherwise
                    figure
            end
            
            plot(self.x, self.y, 'ok', 'markersize', 20)
            title(self.name)
            hold on
            for i = 1 : self.n
                text(self.x(i), self.y(i), sprintf('%d', i), ...
                    'horizontalalignment', 'center', ...
                    'verticalalignment', 'middle')
            end
            d = round(min(pdist([self.x self.y])));
            plot(min(self.x) - d * [1 2], [0 0], 'k', 'linewidth', 4)
            text(min(self.x) - 2 * d, 0, sprintf('\n%d \\mum', d))
            axis equal ij tight off
            ax = axis;
            dx = (ax(2) - ax(1)) / 10;
            dy = (ax(4) - ax(3)) / 10;
            axis(ax + [-dx dx -dy dy])
        end
    end
end
