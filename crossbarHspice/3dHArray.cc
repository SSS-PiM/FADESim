#include "3dHArray.h"

using namespace CBA;

void array3DH::addSelector()
{
    hasSelector = true;
}

void array3DH::removeSelector()
{
    hasSelector = false;
}


bool array3DH::setParas(string s, string t)
{
    stringstream ss;
    ss << t << endl;   
    if (s == "arraySize")
    {
        int m, n, k;
        ss >> n >> m >> k;
        setArraySize(n, m, k);
    }
    else if (s == "selector")
    {
        string str;
        ss >> str;
        if (str == "yes" || str == "y")
        {
            addSelector();
            cout << "selector put into crossbar" << endl;
        }
        else
        {
            removeSelector();
            cout << "selector remove" << endl;
        }
        
    }
    else if (s == "topString")
    {
        gen.push2top(t);
    }
    else if (s == "bottomString")
    {
        gen.push2bot(t);
    }
    else if (s == "line_resistance")
    {
        double x;
        ss >> x;
        for (auto &i : RwlLine)
          for (auto &j : i)
            for (auto &k : j)
                k = x;
        for (auto &i : RblLine)
          for (auto &j : i)
            for (auto &k : j)
                k = x;
    }
    else if (s == "resistance_xyz")
    {
        int x, y, z;
        double r;
        string str;
        ss >> str >> x >> y >> z >> r;
        if (str=="wl" || str=="WL")
            RwlLine.at(x).at(y).at(z) = r;
        else if (str=="bl" || str=="BL")
            RblLine.at(x).at(y).at(z) = r;
        else
        {
            cout << "resistance_xyz commd wrong!" << endl;
            throw 1;
        }
    }
    else if (s == "setUseWordline")
    {
        string dir;
        int h, x;
        ss >> dir >> h >> x;
        assert(dir=="left" || dir=="right");
        if (x!=-1)
        {
            if (dir=="left")
                useLeftWordline.at(h).at(x) = 1;
            else
                useRightWordline.at(h).at(x) = 1;     
        }
        else 
        {
            if (dir=="left")
                for (auto &i : useLeftWordline.at(h))
                    i = 1;
            else
                for (auto &i : useRightWordline.at(h))
                    i = 1;
        }
    }
    else if (s == "setUseBitline")
    {
        string dir;
        int h, y;
        ss >> dir >> h >> y;
        assert(dir=="up" || dir=="down");
        if (y!=-1)
        {
            if (dir=="up")
                useUpBitline.at(h).at(y) = 1;
            else
                useDownBitline.at(h).at(y) = 1; 
        }
        else
        {
            if (dir=="up")
                for (auto &j : useUpBitline.at(h))
                    j = 1;
            else
                for (auto &j : useDownBitline.at(h))
                    j = 1;
        }    
    }
    else if (s == "setNotUseWordline")
    {
        string dir;
        int h, x;
        ss >> dir >> h >> x;
        assert(dir=="left" || dir=="right");
        if (x!=-1)
        {
            if (dir=="left")
                useLeftWordline.at(h).at(x) = 0;
            else
                useRightWordline.at(h).at(x) = 0;     
        }
        else 
        {
            if (dir=="left")
                for (auto &i : useLeftWordline.at(h))
                    i = 0;
            else
                for (auto &i : useRightWordline.at(h))
                    i = 0;
        }
    }
    else if (s == "setNotUseBitline")
    {
        string dir;
        int h, y;
        ss >> dir >> h >> y;
        assert(dir=="up" || dir=="down");
        if (y!=-1)
        {
            if (dir=="up")
                useUpBitline.at(h).at(y) = 0;
            else
                useDownBitline.at(h).at(y) = 0; 
        }
        else
        {
            if (dir=="up")
                for (auto &j : useUpBitline.at(h))
                    j = 0;
            else
                for (auto &j : useDownBitline.at(h))
                    j = 0;
        }    
    }
    else if (s == "setLineV" || s == "setLineV++")
    {
        string type, dir, v;
        int x, h;
        ss >> type >> dir >> h >> x;
        getline(ss, v);
        cout << "setLineV in" << endl;
        assert(type=="wl" || type=="bl" || type=="WL" || type=="BL");
        if (type=="wl" || type=="WL")
            assert(dir=="left" || dir=="right");
        else
            assert(dir=="up" || dir=="down");

        if (type=="wl" || type=="WL")
        {
            if (dir=="left")
            {
                if (x!=-1)
                    VleftWordline.at(h).at(x) = v;
                else 
                    for (auto &i : VleftWordline.at(h))
                        i = v;
            }
            else
            {
                if (x!=-1)
                    VrightWordline.at(h).at(x) = v;
                else 
                    for (auto &i : VrightWordline.at(h))
                        i = v;
            }
        }
        else
        {
            if (dir=="up")
            {
                if (x!=-1)
                    VupBitline.at(h).at(x) = v;
                else
                    for (auto &i : VupBitline.at(h))
                        i = v;
            }
            else
            {
                if (x!=-1)
                    VdownBitline.at(h).at(x) = v;
                else 
                    for (auto &i : VdownBitline.at(h))
                        i = v;
            }
        }
    }
    else if (s == "setCellR")
    {
        int h, x, y, z;
        cout << "setCellR in " << endl;
        ss >> h >> x >> y >> z;
        if (x==-1)
        {
            for (auto &i : arr.at(h))
            {
                if (y==-1)
                    for (auto &j : i)
                        j = z;
                else
                    i.at(y) = z;
            }
        }
        else
        {
            if (y==-1)
                for (auto &j : arr.at(h).at(x))
                    j = z;
            else
                arr.at(h).at(x).at(y) = z;
        }

    }
    else if (s == "senseCellV")
    {
        vector<int> ht, x, y;
        int p;
        int w = 0;
        while (ss >> p)
        {
            if (w==0)
                ht.push_back(p);
            else if (w==1)
                x.push_back(p);
            else
                y.push_back(p);
            w=(w+1)%3;
        }
        if (x.size()!=y.size() || ht.size()!=x.size()) throw 1;
        for (auto &i : ht)
          if (i>=h)
              throw 1;
        for (auto &i : x)
          if (i>=n)
              throw 1;
        for (auto &j : y)
          if (j>=m)
              throw 1;

        for (int i=0; i<x.size(); ++i)
        {
            int nh = ht[i], nx = x[i], ny = y[i];
            gen.push2bot(LogicUnit(UnitType::senseV, {(1+nh)*n*m + nx*m + ny+1}, {}));
            gen.push2bot(LogicUnit(UnitType::senseV, {nh*n*m + nx*m + ny+1}, {}));
        }        
    }
    else if (s == "senseV++")
    {
        vector<int> x;
        int p;
        while (ss >> p)
            x.push_back(p);
        for (auto &i : x)
            gen.push2bot(LogicUnit(UnitType::senseV, {i}, {}));
    }
    else if (s == "senseI++")
    {
        vector<int> x;
        int p;
        while (ss >> p)
            x.push_back(p);
        for (auto &i : x)
            gen.push2bot(LogicUnit(UnitType::senseI, {i, 0}, {}));
    }
    else if (s == "senseWordlineI")
    {
        int h, x;
        int now;
        string dir;
        ss >> dir;
        ss >> h >> x;
        assert(dir=="left" || dir=="right");
        
        if (dir=="left")
            now = getLeft(h, x);
        else
            now = getRight(h, x);
        gen.push2bot(LogicUnit(UnitType::senseI, {now, 0}, {}));
        
    }
    else if (s == "senseBitlineI")
    {
        int h, x;
        int now;
        string dir;
        ss >> dir;
        ss >> h >> x;
        assert(dir=="up" || dir=="down");
        
        if (dir=="up")
            now = getUp(h, x);
        else
            now = getDown(h, x);

        gen.push2bot(LogicUnit(UnitType::senseI, {now, 0}, {}));
        
    }
    else if (s == "build")
    {
        string str;
        ss >> str;
        build(str);
        cout << "build the array to file " << str << endl;
    }
    else
    {
        cout << "wrong command" << endl;
        throw 0;
    }
    return true;
}

int array3DH::getUp(int ht, int y)
{
    int otherPoints = hasSelector? (2*h+1)*m*n : (h+1)*m*n;
    return otherPoints + ht*m + y+1;
}

int array3DH::getLeft(int ht, int x)
{
    int otherPoints = hasSelector? (2*h+1)*m*n : (h+1)*m*n;
    return otherPoints + (h/2+1)*m + ht*n + x+1;
}

int array3DH::getDown(int ht, int y)
{
    int otherPoints = hasSelector? (2*h+1)*m*n : (h+1)*m*n;
    return otherPoints + (h/2+1)*m + (h+1)/2*n + ht*m+y+1;
}

int array3DH::getRight(int ht, int x)
{
    int otherPoints = hasSelector? (2*h+1)*m*n : (h+1)*m*n;
    return otherPoints + (h/2+1)*m*2 + (h+1)/2*n + ht*n + x+1;
}

void array3DH::build(string file)
{
    auto getSel=[&](int ht, int x, int y)->int
    {
        return (h+1)*m*n+ht*m*n+x*m+y+1;
    };
    cout << h << n << m << endl;
    //build ReRAM cells & selectors
    for (int i=0; i<h; ++i)
    {
        for (int j=0; j<n; ++j)
        {
            for (int k=0; k<m; ++k)
            {
                int now = i*m*n + j*m + k+1;
                if (hasSelector)
                {
                    int sel = getSel(i, j, k);
                    gen.push(LogicUnit(UnitType::selector, {now+m*n, sel}, {}));
                    gen.push(LogicUnit(UnitType::ReRAM, {sel, now, arr[i][j][k]}, {}));
                }
                else
                    gen.push(LogicUnit(UnitType::ReRAM, {now, now+n*m, arr[i][j][k]}, {}));
            }
        }
    }

    //build wordlines
    for (int i=0; i<(h+1)/2; ++i)
    {
        for (int j=0; j<n; ++j)
        {
            for (int k=0; k<=m; ++k)
            {
                int pa, pb;
                if (k!=m)
                    pb = (i*2+1)*m*n+j*m+k+1;
                else 
                    pb = getRight(i, j);
                if (k!=0)
                    pa = (i*2+1)*m*n+j*m+k;
                else
                    pa = getLeft(i, j);
                gen.push(LogicUnit(UnitType::linearR, {pa, pb}, { toStr(RwlLine[i][j][k])} ));
            }
        }
    }

    //build bitlines
    for (int i=0; i<h/2+1; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            for (int k=0; k<=n; ++k)
            {
                int pa, pb;
                if (k!=n)
                    pb = 2*i*m*n + k*m + j+1;
                else
                    pb = getDown(i, j);
                if (k!=0)
                    pa = 2*i*m*n+k*m+j+1-m;
                else
                    pa = getUp(i, j);
                gen.push(LogicUnit(UnitType::linearR, {pa, pb}, { toStr(RblLine[i][j][k])}  ));
            }
        }
    }

    //build voltages to the wordlines
    for (int i=0; i<(h+1)/2; ++i)
    {
        for (int j=0; j<n; ++j)
        {
            if (useLeftWordline[i][j])
                gen.push(LogicUnit(UnitType::voltage, {getLeft(i, j)}, { VleftWordline[i][j] } ));
            if (useRightWordline[i][j])
                gen.push(LogicUnit(UnitType::voltage, {getRight(i, j)}, { VrightWordline[i][j] } ));
        }
    }


    //build voltages to the bitlines
    for (int i=0; i<h/2+1; ++i)
    {
        for (int j=0; j<m; ++j)
        {
            if (useUpBitline[i][j])
                gen.push(LogicUnit(UnitType::voltage, {getUp(i, j)}, { VupBitline[i][j] } ));
            if (useDownBitline[i][j])
                gen.push(LogicUnit(UnitType::voltage, {getDown(i, j)}, { VdownBitline[i][j] } ));
        }
    }
    cout << "build over, start print to hspice" << endl;
    cout << "#gen = " << gen.size() << endl;
    ofstream out(file);
    gen.print2Hspice(out);
    cout << "print2 spice over" << endl;

}

