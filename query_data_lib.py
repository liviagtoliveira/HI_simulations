import sqlite3
from math import pi

def getsaxdata(limit):
    """
    will perform query and will readout data
    must choose a limit
    """
    db_path = r"/Volumes/LIVIA/S3SAX01_sqlite3.db"  # path for the db file
    con = sqlite3.connect(db_path)

    # you must create a Cursor object. It will let
    # you execute all the query you need
    cur = con.cursor()

    sel0 = 'SELECT'
    sel1 = ' t1.ra, t1.decl, t1.distance, t1.zapparent, '
    sel2 = ' t1.hiintflux, t1.hilumcenter, t1.hilumpeak, t1.hiwidthpeak, t1.hiwidth50, t1.hiwidth20, t1.diskinclination, '
    sel3 = ' t2.x, t2.y, t2.z, t2.velx, t2.vely, t2.velz from milli_galaxies_line t1, '
    sel4 = ' milli_galaxies_delu t2 where t1.id=t2.id and t1.ra between -0.5 and 0.5 and t1.decl between -0.5 and 0.5 '
    sel5 = ' and t1.zapparent < 1 '
    #sel6 = ' order by t1.hiintflux desc'
    sel7 = f' limit 0,{limit} '
        
    selection = sel0 + sel1 +sel2 + sel3 +sel4 + sel5 + sel7
    
    # Just to clarify which parameter belongs to the profile function
    #
    # DO_lineprofile(x,x0,PHIobs0,PHIobsmax,wobspeak,wobs50,wobs20)
    #
    # hilumcenter  -- PHIobs0
    # hilumpeak    -- PHIobsmax
    # hiwidthpeak  -- wobspeak
    # hiwidth50    -- wobs50
    # hiwidth20    -- wobs20

    # inclination 0 is face on , 180 is edge on

    cur.execute(selection)
    rows = cur.fetchall()
    con.close() # close connection
    
    # print all the first cell of all the rows
    results = [([row[2], row[1], row[0]],
                [row[14], row[15], row[16]],
                row[3],
                [row[3], row[4], row[5], row[6], row[7], row[8], row[9]],
                abs(row[10] * 360.0 / (2 * pi)),)
                for row in rows]
    rphitheta, vpec, zapp, profileval, incl = map(list, zip(*results))
        
    return rphitheta,vpec,zapp,profileval,incl


#rphitheta, vpec, zapp, profileval, incl = getsaxdata(limit=100)
#print(rphitheta) # distance, DEC, RA
#print(vpec) # velocities in x, y, z axes
#print(zapp) # apparent redshift
#print(profileval) # apparent redshift, hiintflux, hilumcenter, hilumpeak, hidiwthpeak, hiwdith50, hiwdith20
#print(incl) # inclination