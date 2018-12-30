using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace DataSorter
{
    class Program
    {
        static List<string> finalCategories;

        static String getCategoryFromLine(string strLine)
        {
            strLine = strLine.Replace("Fld Applictn: ", "");
            strLine = Regex.Replace(strLine, @"[\d_]", string.Empty);
            strLine = strLine.Trim();
            strLine = strLine.Replace("/", "-");
            strLine = strLine.Replace(" ", "_");
            if (strLine == "")
            {
                return "undefined";
            }
            return strLine;
        }

        static String findCategoryDestination(string category, Dictionary<string, long> listOfCategories)
        {
            string fileDestinationPath = "";
            if (category == "")
            {
                fileDestinationPath = "\\undefined";
            }
            else
            {
                long categoryIndex = 0;
                if (listOfCategories.ContainsKey(category))
                {
                    categoryIndex = listOfCategories[category]++;
                }
                else
                {
                    listOfCategories.Add(category, 0);
                }
                switch (categoryIndex % 3)
                {
                    case 0:
                        fileDestinationPath = "\\train";
                        break;
                    case 1:
                        fileDestinationPath = "\\test";
                        break;
                    case 2:
                        fileDestinationPath = "\\validate";
                        break; 
                }
                fileDestinationPath += "\\" + category;
            }
            return fileDestinationPath;
        }

        static void addToDictionary(Dictionary<string, long> dictionary, string phrase)
        {
            if (dictionary.ContainsKey(phrase))
            {
                dictionary[phrase]++;
            }
            else
            {
                dictionary.Add(phrase, 0);
            }
        }

        static int[] createVectorOfCategories(List<string> categories)
        {
            int[] vector = new int[finalCategories.Count];
            foreach (string category in categories)
            {
                if (finalCategories.Contains(category))
                {
                    vector[finalCategories.IndexOf(category)] = 1;
                }
            }
            return vector;
        }

        static void Main(string[] args)
        {
            string filePath = "C:\\Users\\Ada\\Desktop\\TextClassification\\Dane\\Surowe";
            string finalCategoriesPath = "C:\\Users\\Ada\\Desktop\\TextClassification\\DataSorter\\finalCategories.txt";
            string destinationPath = "C:\\Users\\Ada\\Desktop\\TextClassification\\Dane\\Multilabel";

            //var listOfCategories = new Dictionary<string, long>();
            //var countCategory = new Dictionary<string, long>();
            var logFile = File.ReadAllLines(finalCategoriesPath);
            finalCategories = new List<string>(logFile);

            string[] allFiles = Directory.GetFiles(filePath, "*.txt", SearchOption.AllDirectories);

            foreach (var file in allFiles)
            {
                FileInfo info = new FileInfo(file);
                bool isCategory = false;
                String strLine, fileDestinationPath = "";
                var fileContent = new System.Text.StringBuilder();
                string category = "";
                List<string> fileCategories = new List<string>();
                using (StreamReader streamReader = new StreamReader(info.FullName))
                {
                    while (!streamReader.EndOfStream)
                    {
                        strLine = streamReader.ReadLine();
                        if (strLine.Contains("Fld Applictn"))
                        {
                            isCategory = true;
                        }
                        if (strLine.Contains("Program Ref"))
                        {
                            isCategory = false;
                        }
                        if (isCategory)
                        {
                            category = getCategoryFromLine(strLine);
                            if (finalCategories.Contains(category))
                            {
                                fileCategories.Add(category);
                            }
                        }
                        else
                        {
                            fileContent.AppendLine(strLine);
                        }
                    }
                }

                if (fileCategories.Count > 0)
                {
                    string finalDestination = destinationPath + "\\Dane\\";
                    string finalDestination2 = destinationPath + "\\Labelki\\";
                    bool exists = System.IO.Directory.Exists(finalDestination);
                    bool exists2 = System.IO.Directory.Exists(finalDestination2);
                    if (!exists)
                    {
                        System.IO.Directory.CreateDirectory(finalDestination);
                    }
                    if (!exists2)
                    {
                        System.IO.Directory.CreateDirectory(finalDestination2);
                    }
                    System.IO.StreamWriter fileWriter = new System.IO.StreamWriter(finalDestination + info.Name);
                    fileWriter.WriteLine(fileContent.ToString());
                    var labels = createVectorOfCategories(fileCategories);
                    File.WriteAllLines(finalDestination2 + info.Name, labels.Select(x => x.ToString() + "\t").ToArray());
                }
            }
            //Console.WriteLine("Writing to file...");
            //File.WriteAllLines("category.txt", listOfCategories.Where(x=> x.Value > 99).OrderBy(x => x.Key).Select(x =>x.Key).ToArray());
            //File.WriteAllLines("pojedyncze.txt", countCategory.Where(x => x.Value > 99).OrderBy(x => x.Key).Select(x => "[" + x.Key + " " + x.Value + "]").ToArray());

            Console.WriteLine("Work is done");
            Console.Read();
        }
    }
}
