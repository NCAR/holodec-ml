function [updateStruct, dispString] = etd(updateStruct, thisNum, ofNum, everySeconds)
% This function prints out a string to stdout that says how long it will be until
% thisNum == ofNum. The first call should include all the parameters:
% updateStruct = etd(startTime, thisNum, ofNum, everySeconds, dispString, fid)
%
% and succeeding calls could only include:
% updateStruct = etd(updateStruct, thisNum);
% Calling via [updateStruct, dispString] = etd(updateStruct, thisNum);
% returns the string in dispString as opposed to printing it to stdout.

% Version 1.0
% last edited May 5, 2007

    
    if ~isstruct(updateStruct)          % Setup the initial structure
        if nargin < 4
            error('Too few arguments unless the first argument is an update structure');
        end
        temp.startTime      = updateStruct;     % Set the start time
        temp.displayTime    = temp.startTime;   % Initialize the displayTime
        temp.ofNum          = double(ofNum);            % The maximum number we expect
        temp.lastNum        = double(thisNum);
        temp.everySeconds   = double(everySeconds);
        temp.doneTime       = datevec(datenum(temp.startTime + [0 0 0 0 0 etime(clock, temp.startTime)*temp.ofNum/thisNum]));
        temp.firstTime      = true;
        updateStruct        = temp;             % Assign the structure
        clear temp;
    elseif nargin > 3
        error('Too many arguments if first argument ''updateStruct'' ain''t a structure');
    end
    
    if exist('ofNum','var') && ischar(ofNum)
        addString = ofNum;
    else
        addString = '';
    end
    thisNum = double(thisNum);
    if thisNum <= updateStruct.lastNum
%        warning('etd called with an iteration the same or earlier than the last call.');
       return;
    end
    
    % if it hasn't been more than a second yet, return
    c = clock;
    d = updateStruct.displayTime;
    if fix(c(6)) == fix(d(6))
       if nargout > 1
          dispString = [];
       end
       return;
    end
            % If we've elapsed past everySeconds since the last update
    t = etime(c, d);
    if ( t >= updateStruct.everySeconds ...
            || updateStruct.ofNum == thisNum || ...
            (t > 10 && t < updateStruct.everySeconds && updateStruct.firstTime) ) 
            % Calculate the elapsedTime, the estimated until done time, and
            % doneTime.
        elapsedTime = etime(clock, updateStruct.startTime);
        ETD = elapsedTime*updateStruct.ofNum/thisNum;
        doneTime = datenum(updateStruct.startTime + [0 0 0 0 0 ETD]);
            % Calculate the date vectors
        elapsedDateVec = datevec( datenum(date)+datenum([0 0 0 0 0 elapsedTime]));
        ETDDateVec = datevec( datenum(date)+datenum([0 0 0 0 0 ETD]));
        doneDateVec = datevec(doneTime);
        if nargout > 1
            dispString = [sprintf('%d of %d, %05.2f %%%% done, ET: %s, ETD: %s, Done at: %s ', ...
                thisNum, updateStruct.ofNum, thisNum/updateStruct.ofNum*100, ...
                datestr(elapsedDateVec, 13), datestr(ETDDateVec, 13), datestr(doneDateVec,0)),...
                addString 13 10];
        else
            fprintf( [sprintf('%d of %d, %05.2f %%%% done, ET: %s, ETD: %s, Done at: %s ', ...
                thisNum, updateStruct.ofNum, thisNum/updateStruct.ofNum*100, ...
                datestr(elapsedDateVec, 13), datestr(ETDDateVec, 13), datestr(doneDateVec,0)), ...
                addString '\n']);
        end
        updateStruct.displayTime = clock;
        updateStruct.firstTime = false;
        updateStruct.doneTime = doneDateVec;
    else
        if nargout > 1
            dispString = [];
        end
    end
    updateStruct.lastNum = thisNum;
end

