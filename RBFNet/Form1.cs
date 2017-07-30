using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using System.IO;

namespace RBFNet
{
    
    public partial class Form1 : Form
    {
        public static Random rRBF;
        public static Random rLean;
        public static Random r;
        public double[,] WeightsRBF;
        public double[,] WeightsLean;
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            chart1.Series.Clear();
        }

        public void button2_Click(object sender, EventArgs e)
        {
            double y = 0;
            double xn = -1.6;
            double xk = 3.7;
            double xh = 0.3;
            double a = 2;
            int i = 0;
            double max = Double.MinValue;
            double min = Double.MaxValue;
            Series s1 = new Series();
            chart1.Series.Add(s1);
            s1.ChartType = SeriesChartType.Point;
            s1.BorderWidth = 5;
            while (xn <= xk)
            {
                if (xn <= 0) y = Math.Pow(xn, 5) * (1 / Math.Atan(Math.Pow(xn, 3)));
                else if (xn > 0 && xn <= a) y = 5 / (Math.Tan(2 * xn + 3) + 1);
                else y = Math.Pow(xn, 2) * Math.Exp(-xn);
                s1.Points.AddXY(xn, y);
                xn = xn + xh;
                i += 1;
                if (y > max) max = y;
                if (y < min) min = y;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            dataGridView1.Columns.Clear();
            dataGridView1.ColumnCount = 2;
            dataGridView1.Columns[0].Name = "X";
            dataGridView1.Columns[1].Name = "Y";
            rRBF = new Random();
            rLean = new Random();
            double[] input = {-1.6, -1.3, -1, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2, 2.3, 2.6, 2.9, 3.2, 3.5};
            RadialBasNet NET = new RadialBasNet(input);
            double[] output = NET.trainNetRBF();
            WeightsRBF = NET.WeightRBF();
            WeightsLean = NET.WeightLean();
            Series s2 = new Series();
            chart1.Series.Add(s2);
            s2.ChartType = SeriesChartType.Spline;
            s2.BorderWidth = 5;
            for (int i = 0; i < 18; i++)
            {
                s2.Points.AddXY(input[i], output[i]);
                dataGridView1.Rows.Add();
                dataGridView1.Rows[i].Cells[0].Value = input[i];
                dataGridView1.Rows[i].Cells[1].Value = output[i];                
            }
            MessageBox.Show("Обучение завершено!");
            
        }

        private void оПрограммеToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Программа для аппроксимирования на основе радиальной базисной нейроной сети.", "О программе");
        }

        private void button4_Click(object sender, EventArgs e)
        {
            dataGridView1.Columns.Clear();
            dataGridView1.ColumnCount = 2;
            dataGridView1.Columns[0].Name = "X";
            dataGridView1.Columns[1].Name = "Y";
            const int n = 18;
            double[] InNet = new double[n];
            double[] rbfNeuro = new double[n];
            double[] outRBF = new double[n];
            double[] InputMas = { 5.6, 5.9, 6.2, 6.5, 6.8, 7.1, 7.4, 7.7, 8, 8.3, 8.6, 8.9, 9.2, 9.5, 9.8, 10.1, 10.4, 10.7 };
            //double[] InputMas = { 5.6, 6.2, 6.8, 7.4, 8, 8.6, 9.2, 9.8, 10.4, 11, 11.6, 12.2, 12.8, 13.4, 14, 14.6, 15.2, 15.8 };

            NeuronRBF N = new NeuronRBF(InputMas);

            for (int i = 0; i < 18; i++)
            {
                double[] wrbf = new double[n];
                for (int j = 0; j < 18; j++)
                {
                    wrbf[j] = WeightsRBF[i, j];
                }
                double sum = N.Summator(wrbf);
                outRBF[i] = N.FuncActivationRBF(sum);
            }

            NeuronLean M = new NeuronLean(outRBF);

            //линейный слой
            for (int i = 0; i < 18; i++)
            {
                double[] wlean = new double[n];
                for (int j = 0; j < 18; j++)
                {
                    wlean[j] = WeightsLean[i, j];
                }
                double sum = M.Summator(wlean);
                InNet[i] = M.FuncActivationLean(sum);
            }
            chart1.Series.Clear();
            Series s3 = new Series();
            chart1.Series.Add(s3);
            s3.ChartType = SeriesChartType.Spline;
            s3.BorderWidth = 5;
            for (int i = 0; i < 18; i++)
            {
                s3.Points.AddXY(InputMas[i], InNet[i]);
                dataGridView1.Rows.Add();
                dataGridView1.Rows[i].Cells[0].Value = InputMas[i];
                dataGridView1.Rows[i].Cells[1].Value = InNet[i];
            }
        }

        private void выходToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }
    }
}
