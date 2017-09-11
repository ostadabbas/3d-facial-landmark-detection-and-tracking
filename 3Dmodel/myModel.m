clear, close all,

%% Built 3D Model 
load('landmark_result_Yu.txt','r');
landmarks = landmark_result_Yu;

img_w = 602;
img_h = 602;
[ld_m,ld_n] = size(landmarks); % 68 * 2

x = landmarks(:,1);
y = landmarks(:,2);
z = ones(68,1) * 30; %jaw & baseline
z(18:27) = 33; %eyebrew
z(28:31) = 36; %nose2
z(37:48) = 27;  %eye
z(49:68) = 33; %mouth

landmarkTarget = [x,y,z];

%% Rotate
[xrotate,yrotate,zrotate,scale,svName] = rotationInfo(4);
% 1: pitch, 2: yaw, 4:scale

rotateAngle = [xrotate yrotate zrotate scale];
Rot = makehgtform('xrotate',xrotate , 'yrotate', yrotate,'zrotate', zrotate,'scale',scale); 
x4d = [x(:),y(:),z(:),ones(ld_m , 1)]';
x2d = Rot * x4d;
x2 = zeros(ld_m,1); y2 = zeros(ld_m,1); z2 = zeros(ld_m,1);
x2(:) = x2d(1,:)./x2d(4,:);
y2(:) = x2d(2,:)./x2d(4,:);
z2(:) = x2d(3,:)./x2d(4,:);

landmarkSource = [x2,y2,z2];

save(svName, 'landmarkTarget', 'landmarkSource', 'rotateAngle');

function [xrotate,yrotate,zrotate,scale,svName] = rotationInfo(trans)
% xrotate:pitch , yrotate:yaw, zrotate:yaw
if trans == 1 % pitch
    xrotate = pi/4;
    yrotate = 0;
    zrotate = 0;
    scale = 1;
    svName = ('myModel_pitch.mat');
elseif trans == 2 % yaw
    xrotate = 0;
    yrotate = pi/3;
    zrotate = 0;
    scale = 1;
    svName = ('myModel_yaw.mat');
elseif trans == 3 % roll
    xrotate = 0;
    yrotate = 0;
    zrotate = pi/4;
    scale = 1;
    svName = ('myModel_roll.mat');
elseif trans == 4 % scale
    xrotate = 0;
    yrotate = 0;
    zrotate = 0;
    scale = 0.3;
    svName = ('myModel_scale.mat');
end
end


%{
% show rotate figure
figure
%subplot(1,2,2)
mesh(X,Y,Z)
az = 180;
el = 75;
view(az, el);
A       = viewmtx(az,el);


%}


%{
% Projected data
figure,
% Original data
data = get(h,{'XData','YData','Zdata'});
data = [cat(1,data{:})', ones(numel(data{1}),1)];
data_transformed = A*[x, y, z]';
plot(data_transformed(1,:), data_transformed(2,:))


%}
