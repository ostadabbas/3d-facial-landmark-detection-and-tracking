function [] = plotModel(landmarkTarget, landmarkSource, rotateAngle)
x= landmarkTarget(:,1);
y= landmarkTarget(:,2);
z= landmarkTarget(:,3);

x2= landmarkSource(:,1);
y2= landmarkSource(:,2);
z2= landmarkSource(:,3);


xrotate = rotateAngle(1);
yrotate = rotateAngle(2);
zrotate = rotateAngle(3);
scale = rotateAngle(4);
Rot = makehgtform('xrotate',xrotate , 'yrotate', yrotate,...
    'zrotate', zrotate,'scale',scale); % x:pitch , y:yaw


%show raw data
dx=1;
dy=1;
x_edge=[floor(min(x)):dx:ceil(max(x))];
y_edge=[floor(min(y)):dy:ceil(max(y))];
[X,Y]=meshgrid(x_edge,y_edge);
F = TriScatteredInterp(x,y,z);
Z= F(X,Y);

figure()
mesh(X,Y,Z);
rotate3d on;
hold on,
plot(x,y,'ro');
scatter3(x,y,z,20,'r');
title('Initial 3D face mask');
set(gca,'FontSize',14);

% show rotated data
figure()
h = mesh(X,Y,Z);
ax = gca;
t = hgtransform('Parent',ax);
set(h,'Parent',t)
rotate3d on;
set(t,'Matrix',Rot) ;
title('Face with yaw and pitch rotation');
set(gca,'FontSize',14);
hold on,
scatter3(x2,y2, - 80*ones(size(z,1),1),'ro');
scatter3(x2,y2,z2,30,'r');

%figure()
%plot(x,y,'ro');hold on,
%A = plot(x2,y2,'x');
%legend('initial landmarks', 'rotated landmarks');
%set(gca,'FontSize',14);
