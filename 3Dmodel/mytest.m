%% Demo for running the eos fitting from Matlab
 clear,clc,close all
%% Set up some required paths to files:
model_file = '../share/sfm_shape_3448.bin';
blendshapes_file = '../share/expression_blendshapes_3448.bin';
landmark_mappings = '../share/ibug_to_sfm.txt';

%% Load an image and its landmarks in ibug format:
%image = imread('Yu.jpg');
image = imread('../dataSample/Yu.jpg');
load('myModel_yaw.mat');
%landmarks = landmarkSource;
landmarks = landmarkTarget;
%landmarks = load('landmark_result_Yu.txt','r');

image_width = size(image, 2); image_height = size(image, 1);

%% Run the fitting, get back the fitted mesh and pose:
[mesh, render_params] = eos.fitting.fit_shape_and_pose(model_file, blendshapes_file, landmarks, landmark_mappings, image_width, image_height);
% Note: The function actually has a few more arguments to files it
% needs. If you're not running it from within eos/matlab/, you need to
% provide them. See its documentation and .m file.

%% Visualise the fitted mesh using your favourite plot, for example...
figure(1);
plot3(mesh.vertices(:, 1), mesh.vertices(:, 2), mesh.vertices(:, 3), '.');
% or...
FV.vertices = mesh.vertices(:, 1:3);
FV.faces = mesh.tvi;


figure(2);
patch(FV, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong'); light; axis equal; axis off;

%% Visualise the fitting in 2D, on top of the input image:
% Project all vertices to 2D:
render_params.viewport,
render_params.projection,
render_params.modelview,
points_2d = mesh.vertices * (render_params.viewport*render_params.projection*render_params.modelview)';

% Display the image and plot the projected mesh points on top of it:
figure(3);
imshow(image);
hold on;
plot(points_2d(:, 1), points_2d(:, 2), 'g.');
% We can also plot the landmarks the mesh was fitted to:
plot(landmarks(:, 1), landmarks(:, 2), 'ro');


% % landmark index in point_2d
for i = 1:size(landmarks,1)
    [~,I(i)] = min( ((landmarks(i, 1)-points_2d(:, 1)).^2 + (landmarks(i, 2) - points_2d(:, 2)).^2));
end
hold on,
plot(points_2d(I, 1), points_2d(I, 2), 'g.','MarkerSize',30);
% %points_2d(:, 1), points_2d(:, 2), points_2d(:, 3),
% 
figure();
plot3(mesh.vertices(:, 1), mesh.vertices(:, 2), mesh.vertices(:, 3), '.');hold on
scatter3(mesh.vertices(I, 1), mesh.vertices(I, 2), mesh.vertices(I, 3),'r')


landmark3d_ori = [mesh.vertices(I, 1), mesh.vertices(I, 2), mesh.vertices(I, 3)];
save('landmark_3d_ori.mat','landmark3d_ori');
%  landmark3d_trans = [mesh.vertices(I, 1), mesh.vertices(I, 2), mesh.vertices(I, 3)];
%  save('landmark_3d_trans.mat','landmark3d_trans');