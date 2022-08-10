rs.initiate()
var cfg = rs.conf()
cfg.members[0].host="mongo-0.mongo:27017"
rs.reconfig(cfg)
rs.add("mongo-1.mongo:27017")
rs.status()
db.createUser(
{
    user: "docgroup",
    pwd: "docgroup",
    roles: [
            { role: "userAdminAnyDatabase", db: "admin" },
            { role: "readWriteAnyDatabase", db: "admin" },
            { role: "dbAdminAnyDatabase", db: "admin" },
            { role: "clusterAdmin", db: "admin" }
        ]
})