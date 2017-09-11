clear, close all

%% load my model
%load('myModel_combine.mat');
%load('myModel_pitch.mat');
%load('myModel_yaw.mat');
%load('myModel_roll.mat');
%load('myModel_scale.mat');

load('landmark_3d_ori.mat');
load('landmark_3d_trans.mat');
landmarkSource = landmark3d_trans;
landmarkTarget = landmark3d_ori;
figure()
VectorOfHandles(1) = scatter3(landmarkSource(:,1), landmarkSource(:, 2),landmarkSource(:, 3),'x');hold on
VectorOfHandles(2) = scatter3(landmarkTarget(:,1), landmarkTarget(:, 2),landmarkTarget(:, 3), 'o');

%plotModel(landmarkTarget, landmarkSource, rotateAngle);

%% transformation 2D
fixedPoints_2d = getPt(landmarkTarget(:, 1:2));
movingPoints_2d = getPt(landmarkSource(:, 1:2));

tform_2d = fitgeotrans(movingPoints_2d,fixedPoints_2d,'affine');
[transX_2d, transY_2d] = transformPointsForward(tform_2d,...
    landmarkSource(:,1),landmarkSource(:,2)); 


%% transformation 3D

fixedPoints_3d = getPt(landmarkTarget);
fixedPoints_3d = [fixedPoints_3d, ones(3,1)];
movingPoints_3d = getPt(landmarkSource);
movingPoints_3d = [movingPoints_3d, ones(3,1)];

similarity_matrix = movingPoints_3d \ fixedPoints_3d ;
similarity_matrix(:,4)=[0 0 0 1];
tform_3d = affine3d(similarity_matrix);
[transX_3d, transY_3d, transZ_3d] = transformPointsForward(tform_3d,...
    landmarkSource(:,1),landmarkSource(:,2),landmarkSource(:,3));

%% comparing and plotting
figure()
% 2D
subplot(1,2,1)
plot(transX_2d,transY_2d, 'x');
hold on,
plot(landmarkTarget(:, 1), landmarkTarget(:, 2), 'o');
title('Geometric Transformation in 2D Space');
set(gca,'FontSize',14);
legend('Warped landmarks', 'Target landmarks');
%ylim([150 600]);

% 3D
subplot(1,2,2)
%[X, Y, Z] = getXYZ(landmarkTarget(:, 1),landmarkTarget(:, 2),landmarkTarget(:, 3));
%mesh(X,Y,Z);
scatter3(landmarkTarget(:, 1),landmarkTarget(:, 2),landmarkTarget(:, 3));
rotate3d on; hold on,
VectorOfHandles(1) = scatter3(transX_3d,transY_3d,transZ_3d,'x');
VectorOfHandles(2) = scatter3(landmarkTarget(:,1), landmarkTarget(:, 2),landmarkTarget(:, 3), 'o');
title('Geometric Transformation in 3D Space');
set(gca,'FontSize',14);
%legend(VectorOfHandles,{'Warped landmarks', 'Target landmarks'});
%zlim([0 45]);


function [X, Y, Z] = getXYZ(x,y,z)
dx=1;
dy=1;
x_edge=[floor(min(x)):dx:ceil(max(x))];
y_edge=[floor(min(y)):dy:ceil(max(y))];
[X,Y]=meshgrid(x_edge,y_edge);
F = TriScatteredInterp(x,y,z);
Z= F(X,Y);
end


function Pt3 = getPt(landmark)
% size of landmark should be 68 * 2 or 68 * 3.
% outputs are points of middle of left and right eyes, and nose.

eyeL=mean(landmark(37:40,:));
eyeR=mean(landmark(43:46,:));
Pt3 = [eyeL; eyeR; landmark(34,:)];

end
