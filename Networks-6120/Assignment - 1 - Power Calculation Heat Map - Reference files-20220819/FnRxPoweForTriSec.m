function rxPower = FnRxPoweForTriSec(BSTxPwr,BsBw,BsCf,BsHt,BsMBDirn,UeHt,UeBsDist,xdiff,ydiff,PLT)
% Function to calculate the received power at a UE (or a position) from a
% tri-sector Antenna.
%
% Input Parameters
% ----------------
% This function determines the power received at an UE located at
% "BSTxPwr" - The transmission power at BS, ex 43db, 46 db
% "UeBsDist" - distance in meters, from Associated Base Station (AsBS),
% "AzA" - The Azimuth Angle, the angle between the Antenna and the horizon
%         and the UE.
% "HzA" - The Horizontal Angle, the angle between the UE and the direction
%         to the North.
% "MT"  - Mechanical Tilt of the antenna (Deprecated)
% "ET"  - Electrical Tilt of the antenna (Deprecated)
% "BsBw" - bandwith in MHz,  used for transmission by the AsBs
% "PLT" - Path Loss Type. 1 - UrbanMacro/Micro LOS
%                          2 - Urban Macro - NLOS
%                          3 - Urban Micro - NLOS

% Output Parameters
% -----------------
% "RxPower" - Received Power

%%
% Revision history:
% Function done by Mani - March 9 2015
% Updated on 11 Nov 2015. Added input Pwr
% Updated on 26 Mar 2016. Function derived from the base funciton which
% calulated the received power for an omni directional antenna. Now this
% modified function, calculates the received power for a tri-sector
% antenna, which takes into account the Azimuth angle of the antenna,
% and the horizontal angle of the antenna, between the antenna and the UE.

% Reference: N. Tabia, A. Gondran, O. Baala, and A. Caminada, “Interference
% model and evaluation in lte networks,” in 4th Joint IFIP Wireless
% and Mobile Networking Conference (WMNC). IEEE, 2011, pp. 1–6.

% function modified April 15 2016,
% BsMT : Mechanical Tilt of the BS Antenna - DEPRECATED. Currently SINR
% measured only for horizontal position variation. Vertical variation not
% considered. may be in future.
% BsET : Electrical Tilt of the BS Antenna - DEPRECATED. Currently SINR
% measured only for horizontal position variation. Vertical variation not
% considered. may be in future.

%%
% Common Parameters
% Minimum coupling loss in db
MIN_COUPLING_LOSS_DB = 45;

% Relation bewtween the Bandwidth and the number of RB / Channels
if BsBw == 1.25
    RBCount = 6;
elseif BsBw == 3
    RBCount = 15;
elseif BsBw == 5
    RBCount = 25;
elseif BsBw == 10
    RBCount = 50;
elseif BsBw == 15
    RBCount = 75;
elseif BsBw == 20
    RBCount = 100;
else
    disp 'wrong banwidth parameter, Allowed values are 1.25, 3, 5, 10, 15, 20';
    return;
end

% Calculating the Power per channel (or RB), assuming the transmission
% power is equally distributed across the RBs.
BS_CH_POWER_DBM = BSTxPwr - 10*log10(RBCount) ;    % BS Power : 43dBm MANI- Changed 43 to 46 Mar 22 2015

%%
% Pathloss Model: Ref: 3GPP Rel 9, TR 36.814 v9.0.0, 2010-13, Urban Macro BS
%
% Urban Macro and Urban Micro
% LOS
if (PLT == 1)
    pathLoss = 40.0*log10(UeBsDist) + 7.8 - 18.0*log10(BsHt) - 18.0*log10(UeHt) + 2.0*log10(BsCf);
end

% NLOS
%Urban Macro
% PL = 161.04 – 7.1 log10 (W) + 7.5 log10 (h) – (24.37 – 3.7(h/BsHt)2) log10 (BsHt) + (43.42 – 3.1 log10 (BsHt)) (log10 (d)-3) + 20 log10(BsCf) – (3.2 (log10 (11.75 hUT)) 2 - 4.97)
W = 20.0; % road width
h = 50; % average building height

if (PLT == 2)
    pathLoss = 161.04 - (7.1*log10(W)) + (7.5*log10(h)) - ((24.37 - 3.7*(h/BsHt)*(h/BsHt))*log10(BsHt)) + (43.42 - (3.1*log10(BsHt)))*(log10(UeBsDist) - 3) + 20*log10(BsCf) - (3.2*(log10(11.75*UeHt))*(log10(11.75*UeHt)) - 4.97);
end

%NLOS
%Urban Micro
%36.7log10(d) + 22.7 + 26log10(BsCf)
if (PLT == 3)
    pathLoss = 36.7*log10(UeBsDist) + 22.7 + 26*log10(BsCf);
end

% The following is to take care of UE which might be very close to a BS.
% This is the minimum most loss that a UE can expect to have.
% Cross the following assumption and validity.. update the reason.
if(pathLoss < MIN_COUPLING_LOSS_DB)
    pathLoss = MIN_COUPLING_LOSS_DB;
end

%%
% % Logic to determine the vertical angle, i.e the angle subtended between
% % the horizon (horizontal line extended from the atenna a the BS, parallel
% % to the ground) and the line connecting the antenna and the UE. This is
% % the value denoted as Theta(b,t) - Fig 4.a, Reference: Interference model
% % and evaluation in LTE networks, IEEE WMNC 2011.
%
% % Input parameters:
% % h : Height of the BS (Antenna from the ground) in meters
% % d : Distance between the BS and UE in meters.
%
% % Output parameters:
% % VertTheta : Angle in degrees.
%
% %
% % This value should be ideally non zero, since the min UE distance is
% % considered as 35m, i.e. d >= 35m.
% VertTheta = rad2deg(atan(BsHt/UeBsDist));
%
% %
% % Vertical Attenuation : VA
% % Equation 3 in reference
% SLAv = 20;  % 20 dBm
% Th3db = 10; % Vertical Half Power Beam Width (HPBW) - 10 Degrees
%
% % VertTheta - BS.MT - BS.ET
% % (VertTheta - BS.MT - BS.ET)/Th3db
% % ((VertTheta - BS.MT - BS.ET)/Th3db)^2
%
%
% VA = -1 * min((12*((VertTheta - BsMT - BsET)/Th3db)^2),SLAv);

%%
% Logic to determine the Horizontal angle, i.e the angle subtended between
% the direction north and the UE position. This is the value denoted by
% VarPhi(b,t) - Fig 4.b, Reference:
% Interference model and evaluation in LTE networks, IEEE WMNC 2011.

% Input parameters:
% BSx : Base Station X co-ordinate value
% BSy : Base Station Y co-ordinate value
% UEx : User Equipment X co-ordinate value
% UEy : User Equipment Y co-ordinate value
% Output parameters:
% HorzTheta : Angle in degrees.

%
% xdiff = UE.X - BS.X;
% ydiff = UE.Y - BS.Y;


if (xdiff >= 0) && (ydiff > 0)
    % UE in the first quadrant, w.r.to the BS
    %disp('case1');
    HorzTheta = rad2deg(atan(ydiff/xdiff));    %xdiff/ydiff
elseif (xdiff >= 0) && (ydiff < 0)
    % UE in the second quadrant, w.r.to the BS (Going clock wise direction)
    %disp('case2');
    HorzTheta = 270 + rad2deg(atan(xdiff/(-1*ydiff)));    %(-1*ydiff)/xdiff) 90 to 270
elseif (xdiff < 0) && (ydiff < 0)
    % UE in the third quadrant, w.r.to the BS (Going clock wise direction)
    %disp('case3');
    HorzTheta = 180 + rad2deg(atan(ydiff/xdiff));    %(xdiff/ydiff)
elseif (xdiff < 0) && (ydiff > 0)
    % UE in the fourth quadrant, w.r.to the BS (Going clock wise direction)
    HorzTheta = 90 + rad2deg(atan((-1*xdiff)/ydiff));  % 270  % (ydiff/(-1*xdiff)
elseif (xdiff == 0) && (ydiff > 0)
    % UE along +ve y axis
    HorzTheta = 90;
elseif (xdiff == 0) && (ydiff < 0)
    % UE along -ve y axis
    HorzTheta = 270;
elseif (ydiff == 0) && (xdiff > 0)
    % UE along +ve x axis
    HorzTheta = 0;
elseif (ydiff == 0) && (xdiff < 0)
    % UE along -ve x axis
    HorzTheta = 180;
else
    % Boundary case, which should not occur, where diffs are zero
    HorzTheta = 0;
end

% Horizontal Attenuation : HA
% Equation 4 in reference
Am = 25;  % 25 dBm
Phi3db = 70; % Horizontal Half Power Beam Width (HPBW) - 70 Degrees

if BsMBDirn == 90
    if (HorzTheta >=0) && (HorzTheta <= 270)
        HA = -1 * min((12*(((HorzTheta - BsMBDirn)/Phi3db)^2)),Am);
    elseif (HorzTheta >270) && (HorzTheta <= 360)
        HA = -1 * min((12*(((-(360 - HorzTheta) - BsMBDirn)/Phi3db)^2)),Am);
    end
elseif BsMBDirn == 210
    if (HorzTheta >= 0) && (HorzTheta <= 30)
        HA = -1 * min((12*(((((360 + HorzTheta)) - BsMBDirn)/Phi3db)^2)),Am);
    else
        HA = -1 * min((12*(((HorzTheta - BsMBDirn)/Phi3db)^2)),Am);
    end
elseif BsMBDirn == 330
    if (HorzTheta >= 0) && (HorzTheta <= 150)
        HA = -1 * min((12*((((360 + HorzTheta) - BsMBDirn)/Phi3db)^2)),Am);
    else
        HA = -1 * min((12*(((HorzTheta - BsMBDirn)/Phi3db)^2)),Am);
    end
end

%%
%Received Power(dBm) = Transmission Power - Path Loss + Antenna Gain
%(15dbi) + Vertical Attenuation + Horizontal Attenuation.
AG = 15; % Antenna Gain (15dB)

rxPowerDbm = BS_CH_POWER_DBM - pathLoss + HA + AG;  %+ A_PiTh + AG ;%- A_PiTh;% VA - HA;

rxPowerDbW = rxPowerDbm-30;
rxPower = db2pow(rxPowerDbW) ;  % dBW to W


end