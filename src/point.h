class Point {
  public:
    Point();
    Point(int num, float x, float y, float z);
    ~Point() {};
    void draw();
    void print();
    bool isNum(int check);
    void update(int num, float x, float y, float z);
    int _num;
    float _x;
    float _y;
    float _z;
};
