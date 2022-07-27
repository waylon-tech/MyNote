###22.01_IO流(序列流)(了解)
* 1.什么是序列流
	* 序列流可以把多个字节输入流整合成一个, 从序列流中读取数据时, 将从被整合的第一个流开始读, 读完一个之后继续读第二个, 以此类推.
* 2.使用方式
	* 整合两个: SequenceInputStream(InputStream, InputStream)
	* 
			FileInputStream fis1 = new FileInputStream("a.txt");			//创建输入流对象,关联a.txt
			FileInputStream fis2 = new FileInputStream("b.txt");			//创建输入流对象,关联b.txt
			SequenceInputStream sis = new SequenceInputStream(fis1, fis2);	//将两个流整合成一个流
			FileOutputStream fos = new FileOutputStream("c.txt");			//创建输出流对象,关联c.txt
			
			int b;
			while((b = sis.read()) != -1) {									//用整合后的读
				fos.write(b);												//写到指定文件上
			}
			
			sis.close();
			fos.close();

###22.02_IO流(序列流整合多个)(了解)
* 整合多个: SequenceInputStream(Enumeration)
* 
		FileInputStream fis1 = new FileInputStream("a.txt");	//创建输入流对象,关联a.txt
		FileInputStream fis2 = new FileInputStream("b.txt");	//创建输入流对象,关联b.txt
		FileInputStream fis3 = new FileInputStream("c.txt");	//创建输入流对象,关联c.txt
		Vector<InputStream> v = new Vector<>();					//创建vector集合对象
		v.add(fis1);											//将流对象添加
		v.add(fis2);
		v.add(fis3);
		Enumeration<InputStream> en = v.elements();				//获取枚举引用
		SequenceInputStream sis = new SequenceInputStream(en);	//传递给SequenceInputStream构造
		FileOutputStream fos = new FileOutputStream("d.txt");
		int b;
		while((b = sis.read()) != -1) {
			fos.write(b);
		}
	
		sis.close();
		fos.close();

###22.03_IO流(内存输出流*****)(掌握)
* 1.什么是内存输出流
	* 该输出流可以向内存中写数据, 把内存当作一个缓冲区, 写出之后可以一次性获取出所有数据，并不是为了写到硬盘上的某一文件里
* 2.使用方式
	* 创建对象: new ByteArrayOutputStream() 在内存中创建了一个可以增长的内存数组
	* 写出数据: write(int), write(byte[])
	* 获取数据: toByteArray()，toString()
	* 
			FileInputStream fis = new FileInputStream("a.txt");
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			int b;
			while((b = fis.read()) != -1) {
				baos.write(b);
			}
			
			//byte[] newArr = baos.toByteArray();				//将内存缓冲区中所有的字节存储在newArr中
			//System.out.println(new String(newArr));
			System.out.println(baos);
			fis.close();

###22.04_IO流(内存输出流之黑马面试题)(掌握)
* 定义一个文件输入流,调用read(byte[] b)方法,将a.txt文件中的内容打印出来(byte数组大小限制为5)
* 
			FileInputStream fis = new FileInputStream("a.txt");				//创建字节输入流,关联a.txt
			ByteArrayOutputStream baos = new ByteArrayOutputStream();		//创建内存输出流
			byte[] arr = new byte[5];										//创建字节数组,大小为5
			int len;
			while((len = fis.read(arr)) != -1) {							//将文件上的数据读到字节数组中
				baos.write(arr, 0, len);									//将字节数组的数据写到内存缓冲区中
			}
			System.out.println(baos);										//将内存缓冲区的内容转换为字符串打印
			fis.close();
	
###22.05_IO流(对象操作流ObjecOutputStream)(了解)
* 1.什么是对象操作流
	* 该流可以将一个对象写出, 或者读取一个对象到程序中. 也就是执行了序列化和反序列化的操作.
* 2.使用方式
	* 写出: new ObjectOutputStream(OutputStream), writeObject()

			public class Demo3_ObjectOutputStream {
	
				/**
				 * @param args
				 * @throws IOException 
				 * 将对象写出,序列化
				 */
				public static void main(String[] args) throws IOException {
					Person p1 = new Person("张三", 23);
					Person p2 = new Person("李四", 24);
			//		FileOutputStream fos = new FileOutputStream("e.txt");
			//		fos.write(p1);
			//		FileWriter fw = new FileWriter("e.txt");
			//		fw.write(p1);
					//无论是字节输出流,还是字符输出流都不能直接写出对象
					ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("e.txt"));//创建对象输出流
					oos.writeObject(p1);
					oos.writeObject(p2);
					oos.close();
				}
			}

###22.06_IO流(对象操作流ObjectInputStream)(了解)
* 读取: new ObjectInputStream(InputStream), readObject()
	* 
			public class Demo3_ObjectInputStream {

				/**
				 * @param args
				 * @throws IOException 
				 * @throws ClassNotFoundException 
				 * @throws FileNotFoundException 
				 * 读取对象,反序列化
				 */
				public static void main(String[] args) throws IOException, ClassNotFoundException {
					ObjectInputStream ois = new ObjectInputStream(new FileInputStream("e.txt"));
					Person p1 = (Person) ois.readObject();
					Person p2 = (Person) ois.readObject();
					System.out.println(p1);
					System.out.println(p2);
					ois.close();
				}
			
			}
* EOFException 文件读到末尾出现异常
	
###22.07_IO流(对象操作流优化)(了解)
*　将对象存储在集合中写出，即可读取到集合遍历

	Person p1 = new Person("张三", 23);
	Person p2 = new Person("李四", 24);
	Person p3 = new Person("马哥", 18);
	Person p4 = new Person("辉哥", 20);
	
	ArrayList<Person> list = new ArrayList<>();
	list.add(p1);
	list.add(p2);
	list.add(p3);
	list.add(p4);
	
	ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("f.txt"));
	oos.writeObject(list);									//写出集合对象
	
	oos.close();
* 读取到的是一个集合对象

		ObjectInputStream ois = new ObjectInputStream(new FileInputStream("f.txt"));
			ArrayList<Person> list = (ArrayList<Person>)ois.readObject();	//泛型在运行期会被擦除,索引运行期相当于没有泛型
																			//想去掉黄色可以加注解					@SuppressWarnings("unchecked")
			for (Person person : list) {
				System.out.println(person);
			}
		
		ois.close();

###22.08_IO流(加上id号)(了解)
* 注意
	* 要写出的对象必须实现Serializable接口才能被序列化
	* 不用必须加id号

###22.09_IO流(打印流的概述和特点)(掌握)
* 1.什么是打印流
	* 该流可以很方便的将对象的toString()结果输出, 并且自动加上换行, 而且可以使用自动刷出的模式
	* System.out就是一个PrintStream, 其默认向控制台输出信息

			PrintStream ps = System.out;
			ps.println(97);					//其实底层用的是Integer.toString(x),将x转换为数字字符串打印
			ps.println("xxx");
			ps.println(new Person("张三", 23));
			Person p = null;
			ps.println(p);					//如果是null,就返回null,如果不是null,就调用对象的toString()
* 2.使用方式
	* 打印: print(), println()（会查找码表）, write()
	* 自动刷出: PrintStream(OutputStream out, boolean autoFlush)和PrintWriter(OutputStream out, boolean autoFlush) 
	* 注意事项
		* a.自动刷出只对println有用，然而并没有什么用
		* b.打印流只操作数据目的(就只是输出，没有输入)

			PrintWriter pw = new PrintWriter(new FileOutputStream("g.txt"), true);
			pw.write(97);
			pw.print("大家好");
			pw.println("你好");				//自动刷出,只针对的是println方法
			pw.close();

###22.10_IO流(标准输入输出流概述和输出语句)
* 1.什么是标准输入输出流(掌握)
	* System.in是InputStream, 标准输入流, 默认可以从键盘输入读取字节数据（基本io流）
	* System.out是PrintStream, 标准输出流, 默认可以向Console中输出字符和字节数据（打印流，可包装io流）
	* 注意事项
		* 输入流只有单个，要用就不关，因为此流没有连接到硬盘，不会太耗资源
		* 用Scanner封装得到更强大的功能
* 2.修改标准输入输出流(了解)
	* 修改输入流: System.setIn(InputStream)
	* 修改输出流: System.setOut(PrintStream)
	* 
			System.setIn(new FileInputStream("a.txt"));				//修改标准输入流
			//改变后由指向键盘改为指向a.txt
			System.setOut(new PrintStream("b.txt"));				//修改标准输出流
			//改变后由指向控制台改为指向b.txt

			InputStream in = System.in;								//获取标准输入流
			PrintStream ps = System.out;							//获取标准输出流
			int b;
			while((b = in.read()) != -1) {							//从a.txt上读取数据
				ps.write(b);										//将数据写到b.txt上
			}
			
			in.close();
			ps.close();

###22.11_IO流(修改标准输入输出流拷贝图片)(了解)
		System.setIn(new FileInputStream("IO图片.png"));		//改变标准输入流
		System.setOut(new PrintStream("copy.png")); 		//改变标准输出流
		
		InputStream is = System.in;							//获取标准输入流
		PrintStream ps = System.out;						//获取标准输出流
		
		int len;
		byte[] arr = new byte[1024 * 8];
		
		while((len = is.read(arr)) != -1) {
			ps.write(arr, 0, len);
		}
		
		is.close();
		ps.close();

###22.11_IO流(两种方式实现键盘录入)(了解)
* A:BufferedReader的readLine方法。
	* BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
* B:Scanner的nextXxx..()方法
	* Scanner sc = new Scanner(System.in) String line = sc.nextLine();

###22.12_IO流(随机访问流概述和读写数据)(了解)
* A:随机访问流概述
	* RandomAccessFile概述
	* RandomAccessFile类不属于流，是Object类的子类。但它融合了InputStream和OutputStream的功能。
	* 支持对随机访问文件的读取和写入。
* B:read(),write(),seek()
	* 好处：可以用来进行多线程下载

###22.13_IO流(数据输入输出流)(了解)
* 1.什么是数据输入输出流
	* DataInputStream, DataOutputStream可以按照基本数据类型大小读写数据
	* 例如按Long大小写出一个数字, 写出时该数据占8字节. 读取的时候也可以按照Long类型读取, 一次读取8个字节.
* 2.使用方式
	* DataOutputStream(OutputStream), writeInt(), writeLong() 

			DataOutputStream dos = new DataOutputStream(new FileOutputStream("b.txt"));
			dos.writeInt(997);
			dos.writeInt(998);
			dos.writeInt(999);
			
			dos.close();
	* DataInputStream(InputStream), readInt(), readLong()

			DataInputStream dis = new DataInputStream(new FileInputStream("b.txt"));
			int x = dis.readInt();
			int y = dis.readInt();
			int z = dis.readInt();
			System.out.println(x);
			System.out.println(y);
			System.out.println(z);
			dis.close();

###22.14_IO流(Properties的概述和作为Map集合的使用)(了解)
* A:Properties的概述**Map共有功能**
	* Properties 是Hashtable的子类
	* Properties 类表示了一个持久的属性集。
	* Properties 可保存在流中或从流中加载。
	* 属性列表中每个键及其对应值都是一个字符串。 
* B:案例演示
	* Properties作为Map集合的使用
	
###22.15_IO流(Properties的特殊功能使用)(了解)
* A:Properties的		**特殊功能1-集合区**
	* public Object setProperty(String key,String value)
	* public String getProperty(String key)
	* Enumeration<?> propertyNames() 
	* Set<String> stringPropertyNames()
* B:案例演示
	* Properties的特殊功能
	
###22.16_IO流(Properties的load()和store()功能)(了解)
* A:Properties的load()和store()功能 		**特殊功能2-IO区**
* B:案例演示
	* Properties的load()和store()功能
	* 可识别“=”和“:”
	
###22.17_day22总结
* 把今天的知识点总结一遍。
* 一、序列流SequenceInputStream
* 二、内存输出流ByteArrayOutputString
* 三、对象操作流ObjectOutputStream和ObjectInputStream
* 四、打印流printStream和printWriter
* 五、标准输入输出流System.in和System.out
* 六、随机访问流RandomAccessFile
* 七、数据输入输出流DataInputStream和DataOutputStream
* 八、Properties概述及其功能