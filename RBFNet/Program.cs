using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Text;
using System.IO;

namespace RBFNet
{
    static class Program
    {
        /// <summary>
        /// Главная точка входа для приложения.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }

    //Класс нейронов с радиальными базисными активационными функциями
    public class NeuronRBF
    {
        const int nRBF = 18;
        double[] XiRBF = new double[nRBF];
        double[] WeightRBF = new double[nRBF];

        public NeuronRBF(double[] Xi)
        {
            this.XiRBF = Xi;
        }

        public double[] InitWeight()
        {
            Random rRBF = new Random();
            for (int i = 0; i < nRBF; i++)
            {
                WeightRBF[i] = 1 - Form1.rRBF.NextDouble() * 2;
            }
            return WeightRBF;
        }

        public double Summator(double[] WeightRBF)
        {
            double SRBF = 0;
            for (int i = 0; i < nRBF; i++)
            {
                double c = XiRBF[i] * WeightRBF[i];
                SRBF += c;
            }
            return SRBF;
        }

        public double FuncActivationRBF(double x)
        {
            double b = 0.00001;
            double yRBF = Math.Exp(-Math.Pow(x, 2) / 2 * Math.Pow(2 * b, 2));
            return yRBF;
        }
    }

    //Класс нейронов с линейными активационными функциями
    public class NeuronLean
    {
        const int nLean = 18;
        double[] XiLean = new double[nLean];
        double[] WeightLean = new double[nLean];

        public NeuronLean(double[] Xi)
        {
            this.XiLean = Xi;
        }

        public double[] InitWeight()
        {
            Random rLean = new Random();
            for (int i = 0; i < nLean; i++)
            {
                WeightLean[i] = 1 - Form1.rLean.NextDouble() * 2;
            }
            return WeightLean;
        }

        public double Summator(double[] WeightLean)
        {
            double SLean = 0;
            for (int i = 0; i < nLean; i++)
            {
                double c = XiLean[i] * WeightLean[i];
                SLean += c;
            }
            return SLean;
        }

        public double FuncActivationLean(double x)
        {
            double yLean = x;
            return yLean;
        }
    }
    public class RadialBasNet
    {
        
        const int n = 18;
        double[] Xi = new double[n];
        
        public RadialBasNet(double[] Xi)
        {
            this.Xi = Xi;
        }
        double[,] WetRBF = new double[n, n];
        double[,] WetLean = new double[n, n];
        double[] outRBF = new double[n];


        public double[] TargetMas()
        {
            double y = 0;
            double xn = -1.6;
            double xk = 3.7;
            double xh = 0.3;
            double a = 2;
            int i = 0;
            double max = Double.MinValue;
            double min = Double.MaxValue;
            double[] target = new double[18];
            while (xn <= xk)
            {
                if (xn <= 0) y = Math.Pow(xn, 5) * (1 / Math.Atan(Math.Pow(xn, 3)));
                else if (xn > 0 && xn <= a) y = 5 / (Math.Tan(2 * xn + 3) + 1);
                else y = Math.Pow(xn, 2) * Math.Exp(-xn);
                target[i] = xn;
                xn = xn + xh;
                i += 1;
                if (y > max) max = y;
                if (y < min) min = y;
            }
            return target;
        }


        //инициализация и расчет выходного значения сети
        //радиально-базисный слой
        public double[] InitNet()
        {
            double[] rbfNeuro = new double[18];
            double[] leanNeuro = new double[18];
            NeuronRBF n = new NeuronRBF(TargetMas());
            for (int i = 0; i < 18; i++)
            {
                double[] wrbf = n.InitWeight();
                for (int j = 0; j < 18; j++)
                {
                    WetRBF[i, j] = wrbf[j];
                }
                double sum = n.Summator(wrbf);
                outRBF[i] = n.FuncActivationRBF(sum);
            }

            NeuronLean m = new NeuronLean(outRBF);

            //линейный слой
            for (int i = 0; i < 18; i++)
            {
                double[] wlean = m.InitWeight();
                for (int j = 0; j < 18; j++)
                {
                    WetLean[i, j] = wlean[j];
                }
                double sum = m.Summator(wlean);
                leanNeuro[i] = m.FuncActivationLean(sum);
            }
            return leanNeuro;
        }

        //метод для обучения нейросети
        public double[] trainNetRBF()
        {
            int c = 0;
            double functarget = 0;
            double[] errLean = new double[n];
            double[] errRBF = new double[n];
            double err = 0.1;
            double LeanRate = 0.1;
            double[] net = new double[n];
            double[] target = {7.88, 3.24, 1.27, 0.5, 0.16, 0.01, 3.95, 2.32, 0.5, -5.65, 10.52, 4.47, 2.67, 0.53, 0.5, 0.46, 0.42, 0.37};
            double[] inNet = InitNet(); 
            double[] WeigtDeltaRBF = new double[n];
            double[] WeigtDeltaLean = new double[n];
            double[] LeanFuncdx = new double[n];
            double[] RBFFuncdx = new double[n];

            for (int i = 0; i < 18; i++)
            {

                double f = 0;
                f = Math.Pow((inNet[i] - target[i]), 2);
                functarget = functarget + f;
            }
            functarget = functarget * 0.5;
            string fun;
            fun = Convert.ToString(functarget);
            //MessageBox.Show(fun, "Ошибка при инциализации сети");

            while (functarget > err)
            {
                //расчет ошибки  и изменение весов для выходного линейного слоя сети
                for (int i = 0; i < n; i++)
                {
                    errLean[i] = inNet[i] - target[i];
                    LeanFuncdx[i] = 1;
                    WeigtDeltaLean[i] = errLean[i] * LeanFuncdx[i];
                    for (int j = 0; j < n; j++)
                    {
                        WetLean[i, j] = WetLean[i, j] - outRBF[i] * WeigtDeltaLean[i] * LeanRate;
                    }
                }

                //расчет ошибки  и изменение весов для скрытого радиального слоя сети
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        errRBF[i] = errRBF[i] + WetLean[i, j] * WeigtDeltaLean[i];
                    }

                    RBFFuncdx[i] = outRBF[i];
                    WeigtDeltaRBF[i] = errRBF[i] * RBFFuncdx[i];
                    for (int j = 0; j < n; j++)
                    {
                        WetRBF[i, j] = WetRBF[i, j] - target[i] * WeigtDeltaRBF[i] * LeanRate;
                    }
                }

                //-------------------------------------------------------------

                //инициализация сети с использованием случайно изменнных весов
                double[] rbfNeuro = new double[n];
                NeuronRBF N = new NeuronRBF(TargetMas());

                for (int i = 0; i < 18; i++)
                {
                    double[] wrbf = new double[n];
                    for (int j = 0; j < 18; j++)
                    {
                        wrbf[j] = WetRBF[i, j];
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
                        wlean[j] = WetLean[i, j];
                    }
                    double sum = M.Summator(wlean);
                    inNet[i] = M.FuncActivationLean(sum);
                }

                // -------------------------------------------
                
                //рассчет целевой функции
                for (int i = 0; i < 18; i++)
                {
                    
                    double f = 0;
                    f = Math.Pow((inNet[i] - target[i]), 2);
                    functarget = functarget + f;
                }
                functarget = functarget * 0.5;
                c++;
                //fun = Convert.ToString(functarget);
                //MessageBox.Show(fun, "Значение ошибки на этапе обучения");
            }
            
            return inNet;
        }

        public double[,] WeightRBF()
        {
            return WetRBF;
        }

        public double[,] WeightLean()
        {
            return WetLean;
        }
    }
}
