%% Test case 1: Void in Uniform Tensile Field

%% Result %%%%

filename = 'case1-result';

load(filename)
figure; hold on
plot(X{1}, Y{1}, 'DisplayName', 'Analytical Solution');
scatter(X{2}, Y{2}, 's', 'filled', 'DisplayName', 'Finite Element Solution')
plot(X{3}, Y{3}, 'o', 'DisplayName', 'Spherical Harmonics Solution')
leg = legend('show'); ax = gca;

xlim([1, 3]); ylim([1, 2.2]);
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
xlabel('x/r_0', 'FontSize', 24);
ylabel('\sigma_z/S_0', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Test case 2: Void-Dislocation Interaction

%% Result for different mode # %%%%

filename = 'case2-result';

load(filename)
figure; hold on
plot(X{4}, Y{4}, '*', 'DisplayName', 'l_{max}=5')
plot(X{3}, Y{3}, '*', 'DisplayName', 'l_{max}=10')
plot(X{2}, Y{2}, '*', 'DisplayName', 'l_{max}=15')
plot(X{1}, Y{1}, '*', 'DisplayName', 'l_{max}=20')
plot(X{5}, Y{5}, 'DisplayName', 'Analytical Solution')
leg = legend('show', 'Location', 'southeast'); ax = gca;

xlim([0, 3]); %ylim([-0.035, 0]);
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
xlabel('z/r_0', 'FontSize', 24);
ylabel('\sigma_{yz}/\mu', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Error Analysis for different mode # %%%%

filename = 'case2-err';

load(filename)
figure; hold on
yyaxis right
plot(X{2}, Y{2}, 'x:', 'DisplayName', 'Matrix Solving Time (s)')
plot(X{3}, Y{3}, 'x-', 'DisplayName', 'Solution Reconstructing Time (s)')
ylabel('Time (s)', 'FontSize', 24);
yyaxis left
plot(X{1}, Y{1}*100, '*--',  'DisplayName', 'Maximum Relative Error')

leg = legend('show', 'Location', 'north'); ax = gca;
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis(1).FontSize = 14;
ax.YAxis(2).FontSize = 14;
xlabel('l_{max}', 'FontSize', 24);
ylabel('Relative Error (%)', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Result for different distance t %%%%

filename = 'case2-result-ts';

load(filename)
figure; hold on
plot(X{5}, Y{5}, '*-', 'DisplayName', 't/r_0=1.2')
plot(X{4}, Y{4}, '*-', 'DisplayName', 't/r_0=1.4')
plot(X{3}, Y{3}, '*-', 'DisplayName', 't/r_0=1.6')
plot(X{2}, Y{2}, '*-', 'DisplayName', 't/r_0=1.8')
plot(X{1}, Y{1}, '*-', 'DisplayName', 't/r_0=2.0')
leg = legend('show', 'Location', 'southeast'); ax = gca;

xlim([0, 3]); %ylim([-0.035, 0]);
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
xlabel('z/r_0', 'FontSize', 24);
ylabel('\sigma_{yz}/\mu', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Error Analysis for different distance t %%%%

filename = 'case2-error-ts';

load(filename)
figure
yyaxis left
plot(X{1}, Y{1}, 'x-', 'DisplayName', 'Non-zero Coefficients')
ylabel('Number of Non-zero SH Coefficients', 'FontSize', 12);
yyaxis right
semilogy(X{2}, Y{2}, '*--', 'DisplayName', 'Maximum Relative Error')

leg = legend('show', 'Location', 'northeast'); ax = gca;
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis(1).FontSize = 14;
ax.YAxis(2).FontSize = 14;
xlabel('t/r_0', 'FontSize', 24);
ylabel('Relative Error', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Compare with FEM %%%%

filename = 'case2-FEM';

load(filename)
figure; hold on
plot(X{1}, Y{1}, 'x-', 'DisplayName', 'Spherical Harmonics Solution')
plot(X{2}, -Y{2}, '*', 'DisplayName', 'Finite Element Solution')

leg = legend('show', 'Location', 'southeast'); ax = gca;
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
xlabel('x/r_0', 'FontSize', 24);
ylabel('\sigma_{yz}/\mu', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Test case 3: Prismatic Dislocation Loop

%% Results for the loops smaller than the void %%%%

filename = 'case3-result-small';

load(filename)
figure; hold on
plot(z_list, fz_40, 'x-', 'DisplayName', 'r_0/b = 40')
plot(z_list, fz_200, 'x-', 'DisplayName', 'r_0/b = 200')
plot(z_list, fz_400, 'x-', 'DisplayName', 'r_0/b = 400')
plot(z_list, fz_4000, 'x-', 'DisplayName', 'r_0/b = 4000')
plot(z_list, fz_40000, 'x-', 'DisplayName', 'r_0/b = 40000')
xlim([0, 4])
ylim([-0.02, 0])
annotation('textarrow', [0.45, 0.4], [0.7, 0.8],...
    'String', 'Increasing r_0', 'FontSize', 14)

leg = legend('show', 'Location', 'southeast'); ax = gca;
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
xlabel('z/r_0', 'FontSize', 24);
ylabel('f_{z}/b\mu', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')

%% Results for the loops larger than the void %%%%

filename = 'case3-result-large';

load(filename)
figure; hold on
plot(z_list, fg, 'x-', 'DisplayName', 'f_g')
plot(z_list, fc, 'o-', 'DisplayName', 'f_c')
xlim([-4, 4])
%ylim([-0.014, 0.002])

leg = legend('show', 'Location', 'southeast'); ax = gca;
leg.FontSize = 14;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
xlabel('z/r_0', 'FontSize', 24);
ylabel('f_{z}/b\mu', 'FontSize', 24);

saveas(gca, filename, 'epsc')
saveas(gca, filename, 'png')
