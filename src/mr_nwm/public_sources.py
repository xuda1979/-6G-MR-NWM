"""Pinned, measured public dataset identities; raw measurements are not vendored."""

UCC4G_URL = "https://zenodo.org/api/records/1219679/files/LTE_Dataset.zip/content"
UCC4G_SHA256 = "45e95f8a6934da230067803983b45b20e5dfb0179431d43902c8193db770e6bd"
UCC5G_REVISION = "306d92eba81cb4f1195d8b8eba625666a02a10ec"
UCC5G_URL = (
    "https://raw.githubusercontent.com/uccmisl/5Gdataset/"
    + UCC5G_REVISION
    + "/5G-production-dataset.zip"
)
UCC5G_SHA256 = "abc729d696b5e0ba34a9c6ed35a851f2868d2b5071be5efb8a49691c23bb1a9b"
CICV_REVISION = "0ad0b8307b918b22754ff651f750e70c7960abd2"
# Relative path, Git blob identity, exact published byte count. This list is
# frozen from the pinned tree; aggregate all.txt and spreadsheets are excluded.
CICV_FILES = (
    (
        "data/Arterial road/n78/50/arterial_n78_v50_run01.txt",
        "d665a04ec9f5353e6297147b8a4b035b5d7c9ecf",
        258413,
    ),
    (
        "data/Arterial road/n78/50/arterial_n78_v50_run02.txt",
        "fd09d519bbc5c3c94ac671a8fd03556c9b5fa9b8",
        246730,
    ),
    (
        "data/Arterial road/n78/50/arterial_n78_v50_run03.txt",
        "d665a04ec9f5353e6297147b8a4b035b5d7c9ecf",
        258413,
    ),
    (
        "data/Arterial road/n78/60/arterial_n78_v60_run01.txt",
        "9c3b6e5c0a10aad7f2d797150b4fc924f358fa2a",
        220460,
    ),
    (
        "data/Arterial road/n78/60/arterial_n78_v60_run02.txt",
        "6970fccb1ef8993f7f9cff84e74fc450556fe8fb",
        215948,
    ),
    (
        "data/Arterial road/n78/70/arterial_n78_v70_run01.txt",
        "60e2b554255f647133562b0bcb81da8b9c632f8f",
        187391,
    ),
    (
        "data/Arterial road/n78/70/arterial_n78_v70_run02.txt",
        "afc6131d5526397deb7872c3dbe71f9f3b3e8c79",
        198749,
    ),
    (
        "data/Arterial road/n78/80/arterial_n78_v80_run01.txt",
        "da80374d1810bd447fab03779a7293cf63791393",
        193028,
    ),
    (
        "data/Arterial road/n78/80/arterial_n78_v80_run02.txt",
        "80c4f5636773347e1047c6bbfca5d279230d0aef",
        202372,
    ),
    (
        "data/Arterial road/n8/v50/arterial_n8_v50_run01.txt",
        "b83a8da77281f3d1778c5b18dc626cb45e46a128",
        144608,
    ),
    (
        "data/Arterial road/n8/v50/arterial_n8_v50_run02.txt",
        "6b22068995646e35756f4be99599bd7267f6249d",
        112192,
    ),
    (
        "data/Arterial road/n8/v50/arterial_n8_v50_run03.txt",
        "5b922618a8edba12513ca790b4f08350c4d0517c",
        117895,
    ),
    (
        "data/Arterial road/n8/v50/arterial_n8_v50_run04.txt",
        "ff205f29a9fcae46b5b74b3b5e5e26f5b013b5ed",
        120795,
    ),
    (
        "data/Arterial road/n8/v50/arterial_n8_v50_run05.txt",
        "dc977b12ed3e1f5740fa5fda76e81e88bf84aec4",
        122853,
    ),
    (
        "data/Arterial road/n8/v60/arterial_n8_v60_run01.txt",
        "12246855652642d499960d9193cc90a245d14018",
        87041,
    ),
    (
        "data/Arterial road/n8/v60/arterial_n8_v60_run02.txt",
        "c0ba9f8781c83cfb376e88bf21112234c5253a7a",
        64174,
    ),
    (
        "data/Arterial road/n8/v60/arterial_n8_v60_run03.txt",
        "67e777746f11a9c1ad48954caa53349d7b7fb0a4",
        116698,
    ),
    (
        "data/Arterial road/n8/v60/arterial_n8_v60_run04.txt",
        "71ce4cce9171c4537c9a2883ee1b9d6eba7e56c6",
        88616,
    ),
    (
        "data/Arterial road/n8/v60/arterial_n8_v60_run05.txt",
        "b579b121c67b84aab298b1f688c6d3fa569e6a58",
        117711,
    ),
    (
        "data/Arterial road/n8/v70/arterial_n8_v70_run01.txt",
        "5b4ed68c5f7eadf60fee4aba7231ee7cfc6ef7a9",
        101062,
    ),
    (
        "data/Arterial road/n8/v70/arterial_n8_v70_run02.txt",
        "775afc0fbec5065222fb124b3cc03307b506704a",
        86032,
    ),
    (
        "data/Arterial road/n8/v70/arterial_n8_v70_run03.txt",
        "e0ac5d3e0e191a643564e24f73751db49f432d55",
        95853,
    ),
    (
        "data/Arterial road/n8/v70/arterial_n8_v70_run04.txt",
        "2f285521e8c5ac9fc4fb55c162d8eded3ae40d95",
        93049,
    ),
    (
        "data/Arterial road/n8/v70/arterial_n8_v70_run05.txt",
        "1ef407bfa42e4d910397aff2c531d0df4dad74f1",
        92051,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run01.txt",
        "7cb26e4ac312ba4789fa640196909462c2e04b4d",
        87268,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run02.txt",
        "1506f08382213e553cde002e11105432fef5b62d",
        91149,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run03.txt",
        "0c938aec5161ca0c8b5db6c9f4a8271dfa2d352d",
        94546,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run04.txt",
        "4b3ff97988fab8645786055cab6f5790bb740a35",
        84799,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run05.txt",
        "b5aeb8db3b0049fd5b0e508534612919da202564",
        84468,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run06.txt",
        "44fc263188d48aaa66aa2966dad338dbf17c376e",
        84447,
    ),
    (
        "data/Arterial road/n8/v80/arterial_n8_v80_run07.txt",
        "562927f08cb3e2ed5290930e3074826788ed829b",
        83453,
    ),
    (
        "data/Urban road/1-n8/v0/urban_n8_v0_run01.txt",
        "6c5b0f8e5a2c4b2a50d619a7af6f367d03415c6e",
        114777,
    ),
    (
        "data/Urban road/1-n8/v0/urban_n8_v0_run02.txt",
        "3dc631908d53f7d0dbc74100537ddf60f18e1836",
        103753,
    ),
    (
        "data/Urban road/1-n8/v0/urban_n8_v0_run03.txt",
        "a1f6e645e7a896318876cdd7d419af2d458edf7f",
        101948,
    ),
    (
        "data/Urban road/1-n8/v20/urban_n8_v20_run01.txt",
        "723a68ec4d66f70a83c4fe6a429c90db9c0d7f8d",
        589289,
    ),
    (
        "data/Urban road/1-n8/v20/urban_n8_v20_run02.txt",
        "a70d487e54616c52ed326ebdfa0131a58a9c7f04",
        574524,
    ),
    (
        "data/Urban road/1-n8/v20/urban_n8_v20_run03.txt",
        "62fccec56c610347e83f70b88d8426156c4a63b8",
        601409,
    ),
    (
        "data/Urban road/1-n8/v30/urban_n8_v30_run01.txt",
        "690171da28fc50c1abd8d83f12ed2a26fe8d9afe",
        424487,
    ),
    (
        "data/Urban road/1-n8/v30/urban_n8_v30_run02.txt",
        "bb28e6fa2e95d11c755cd9ba7308078baa157b58",
        411762,
    ),
    (
        "data/Urban road/1-n8/v30/urban_n8_v30_run03.txt",
        "4bc0a4d08cceb92f97acb8456adf311c35b765b0",
        387049,
    ),
    (
        "data/Urban road/1-n8/v40/urban_n8_v40_run01.txt",
        "d125ed7c49c5ec49c3de34acacfb119940847feb",
        314995,
    ),
    (
        "data/Urban road/1-n8/v40/urban_n8_v40_run02.txt",
        "8f45356cf76fc0be88c67ffd79d8a6fc02abe084",
        340679,
    ),
    (
        "data/Urban road/1-n8/v40/urban_n8_v40_run03.txt",
        "386042ac2d5b25e64c905e666a9c976ed7f45ede",
        355890,
    ),
    (
        "data/Urban road/2-n78/v0/urban_n78_v0_run01.txt",
        "c75deddf421858763ad3c42d5d4364d436f21dc9",
        105135,
    ),
    (
        "data/Urban road/2-n78/v0/urban_n78_v0_run02.txt",
        "1ec82f2d9c98a68ae4212fc05b0b3e5649e256f2",
        107248,
    ),
    (
        "data/Urban road/2-n78/v0/urban_n78_v0_run03.txt",
        "91d661662a1293595fa7226235d371bd458f1dea",
        91884,
    ),
    (
        "data/Urban road/2-n78/v20/urban_n78_v20_run01.txt",
        "fe0d3f2524829a4f6945627cf59cc6fb068e061a",
        599923,
    ),
    (
        "data/Urban road/2-n78/v20/urban_n78_v20_run02.txt",
        "dffda9ba27a80c73e68b3e1f4529465f8d412b72",
        606098,
    ),
    (
        "data/Urban road/2-n78/v20/urban_n78_v20_run03.txt",
        "fb21a5c7a4ca68349aab18b2311681e0bf3b77aa",
        641850,
    ),
    (
        "data/Urban road/2-n78/v30/urban_n78_v30_run01.txt",
        "f936216d03a3a6287ce453c709486c20d06c7d08",
        414250,
    ),
    (
        "data/Urban road/2-n78/v30/urban_n78_v30_run02.txt",
        "8a1e1a75ccdcf59b789fd1363cc4731c90b9cedd",
        404190,
    ),
    (
        "data/Urban road/2-n78/v30/urban_n78_v30_run03.txt",
        "07712ec9d3cd23e194e1835f5cace3ddd87a0ede",
        382333,
    ),
    ("data/W2S/n78/V30/w2s_n78_v30_run01.txt", "560aef3469e2d2a1d97c3b2295fd87ee81f980b3", 299126),
    ("data/W2S/n78/V30/w2s_n78_v30_run02.txt", "cd013b8cfd643cfe1e90f3351136ee00e712801c", 305801),
    ("data/W2S/n78/V30/w2s_n78_v30_run03.txt", "e1d920be7fb9f040806d84aefebeeeb20110cf30", 307191),
    ("data/W2S/n78/V30/w2s_n78_v30_run04.txt", "b74ff1330b2c023d9ce4afcac721317e845059fa", 250032),
    ("data/W2S/n78/V30/w2s_n78_v30_run05.txt", "8b5a54d3bb05469e2a928813da7300062f67bea1", 241759),
    ("data/W2S/n78/V30/w2s_n78_v30_run06.txt", "324dc83ecb1bd118c0cdff0d9e0d170df56e0f9d", 238346),
    ("data/W2S/n78/V50/s2w_n78_v50_run01.txt", "f9af045fd0e37ba2ce802f08484073e591cb5095", 120717),
    ("data/W2S/n78/V50/s2w_n78_v50_run02.txt", "343abc856c412c0dc61afb9b3a1617bbb99eed52", 56563),
    ("data/W2S/n78/V50/s2w_n78_v50_run03.txt", "403b706be63f4e3f18a12c4e5b0c17117c0f2fd6", 162423),
    ("data/W2S/n78/V50/s2w_n78_v50_run04.txt", "f4bab745e7593f97ec06e82e8b0a050e5fc42873", 110503),
    ("data/W2S/n78/V50/s2w_n78_v50_run05.txt", "e3210301f44aa519735d2c5595d40236db6c87e9", 186854),
    ("data/W2S/n78/V50/w2s_n78_v50_run01.txt", "1898aa063947aeafae0fdd75c493525eb9dd5bf3", 122328),
    ("data/W2S/n78/V50/w2s_n78_v50_run02.txt", "3eface1e0403f3fad52c56d277df663d78e99789", 201645),
    ("data/W2S/n78/V50/w2s_n78_v50_run03.txt", "5431e90581ad250a1dc72918710372642494209f", 195884),
    ("data/W2S/n78/V50/w2s_n78_v50_run04.txt", "4cb314300da7da46143123383ba38b6549725552", 170373),
    ("data/W2S/n78/V50/w2s_n78_v50_run05.txt", "58e07f39f08d0d2570161d3b091db80d5acdf4af", 166728),
    ("data/W2S/n78/V50/w2s_n78_v50_run06.txt", "afb7aa358eec1c91a5f7dc616f2f9c8e94d5edb4", 165654),
    ("data/W2S/n8/V30/s2w_n8_v30_run01.txt", "43d4016d981c88c347eab096c90e873da293ac5d", 280528),
    ("data/W2S/n8/V30/s2w_n8_v30_run02.txt", "a0394f616c22d54603ebc8a51af9999c7e77a5d5", 293096),
    ("data/W2S/n8/V30/s2w_n8_v30_run03.txt", "0c4f56f9aa2b2461666740feeef0f94c5f57f32f", 283485),
    ("data/W2S/n8/V30/w2s_n8_v30_run01.txt", "af6f308ea75d04611cc6377eb4802dc4e3686907", 314339),
    ("data/W2S/n8/V30/w2s_n8_v30_run02.txt", "b4426ce056c093337620dae04a1db35daa26da79", 271352),
    ("data/W2S/n8/V30/w2s_n8_v30_run03.txt", "c6b09ef43fe7bb1dd69bbd9023641bd9ba3be159", 268481),
    ("data/W2S/n8/V30/w2s_n8_v30_run04.txt", "2add7c0baf12a35dd983020a651f61faaa6cca0e", 252689),
    ("data/W2S/n8/V30/w2s_n8_v30_run05.txt", "e6234d8ac6eb445e9b6df545b8f6dc292b416013", 248303),
    ("data/W2S/n8/V30/w2s_n8_v30_run06.txt", "cd00547f5f62841b6d018c2ad2e7548b75f7a996", 256436),
    ("data/W2S/n8/V30/w2s_n8_v30_run07.txt", "ecab57a3948f4d685525ea6d8c142b52faf0b093", 257878),
    ("data/W2S/n8/V30/w2s_n8_v30_run09.txt", "91d649eb084a0917e996ae35cd92fcdfe4bfeebe", 254811),
    ("data/W2S/n8/V30/w2s_n8_v30_run10.txt", "d4ca1171cef98fde194f87e295bd100fc6868482", 277148),
    ("data/W2S/n8/V50/s2w_n8_v50_run01.txt", "fdd9e3c05845d9020520c41a473bbba1d16a00b0", 179225),
    ("data/W2S/n8/V50/s2w_n8_v50_run02.txt", "015262fccfe809a4d5786ee62d92e8f3dfaafd85", 182885),
    ("data/W2S/n8/V50/s2w_n8_v50_run03.txt", "c44b2bb2ff7b978001aa2bd939f21ae78df0f3f6", 159923),
    ("data/W2S/n8/V50/w2s_n8_v50_run01.txt", "9eb043ee481c1e157d157555f56b08cef7e18acb", 222838),
    ("data/W2S/n8/V50/w2s_n8_v50_run02.txt", "2fe9fcb92c5c89591f68df9803668924695e8600", 227817),
    ("data/W2S/n8/V50/w2s_n8_v50_run03.txt", "a8398986b4e73410c7d27c0f491ba146b2d8b656", 209035),
    ("data/W2S/n8/V50/w2s_n8_v50_run04.txt", "0316891519be4c2f66f59700b93cb661aa1c9eac", 112000),
    ("data/W2S/n8/V50/w2s_n8_v50_run05.txt", "e7038e6df44505c4e85ee784a8fba95f90b395f8", 101694),
    ("data/W2S/n8/V50/w2s_n8_v50_run06.txt", "1f3a4ef82a4bf762d8caa3bb6973bcd400b2c902", 107929),
)
