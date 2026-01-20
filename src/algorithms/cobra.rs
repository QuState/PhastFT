//! This module provides several implementations of the bit reverse permutation, which is
//! essential for algorithms like FFT.
//!
//! In practice, most FFT implementations avoid bit reversals; however this comes at a computational
//! cost as well. For example, Bailey's 4 step FFT algorithm is O(N * lg(N) * lg(lg(N))).
//! The original Cooley-Tukey implementation is O(N * lg(N)). The extra term in the 4-step algorithm
//! comes from incorporating the bit reversals into each level of the recursion. By utilizing a
//! cache-optimal bit reversal, we are able to avoid this extra cost [1].
//!
//! # References
//!
//! [1] L. Carter and K. S. Gatlin, "Towards an optimal bit-reversal permutation program," Proceedings 39th Annual
//! Symposium on Foundations of Computer Science (Cat. No.98CB36280), Palo Alto, CA, USA, 1998, pp. 544-553, doi:
//! 10.1109/SFCS.1998.743505.
//! keywords: {Read-write memory;Costs;Computer science;Drives;Random access memory;Argon;Registers;Read only memory;Computational modeling;Libraries}

const BLOCK_WIDTH: usize = 128; // size of the cacheline
const LOG_BLOCK_WIDTH: usize = 7; // log2 of cacheline

/// Fully unrolled bit reversal for size 64 (log_n = 6)
fn bit_rev_64<T>(buf: &mut [T]) {
    // Pre-computed bit-reversed pairs for N=64
    // Only swap when i < rev(i) to avoid double swapping
    buf.swap(1, 32);
    buf.swap(2, 16);
    buf.swap(3, 48);
    buf.swap(4, 8);
    buf.swap(5, 40);
    buf.swap(6, 24);
    buf.swap(7, 56);
    buf.swap(9, 36);
    buf.swap(10, 20);
    buf.swap(11, 52);
    buf.swap(13, 44);
    buf.swap(14, 28);
    buf.swap(15, 60);
    buf.swap(17, 34);
    buf.swap(19, 50);
    buf.swap(21, 42);
    buf.swap(22, 26);
    buf.swap(23, 58);
    buf.swap(25, 38);
    buf.swap(27, 54);
    buf.swap(29, 46);
    buf.swap(31, 62);
    buf.swap(35, 49);
    buf.swap(37, 41);
    buf.swap(39, 57);
    buf.swap(43, 53);
    buf.swap(47, 61);
    buf.swap(55, 59);
}

/// Fully unrolled bit reversal for size 128 (log_n = 7)
fn bit_rev_128<T>(buf: &mut [T]) {
    // Pre-computed bit-reversed pairs for N=128
    // Only swap when i < rev(i) to avoid double swapping
    buf.swap(1, 64);
    buf.swap(2, 32);
    buf.swap(3, 96);
    buf.swap(4, 16);
    buf.swap(5, 80);
    buf.swap(6, 48);
    buf.swap(7, 112);
    buf.swap(9, 72);
    buf.swap(10, 40);
    buf.swap(11, 104);
    buf.swap(12, 24);
    buf.swap(13, 88);
    buf.swap(14, 56);
    buf.swap(15, 120);
    buf.swap(17, 68);
    buf.swap(18, 36);
    buf.swap(19, 100);
    buf.swap(21, 84);
    buf.swap(22, 52);
    buf.swap(23, 116);
    buf.swap(25, 76);
    buf.swap(26, 44);
    buf.swap(27, 108);
    buf.swap(29, 92);
    buf.swap(30, 60);
    buf.swap(31, 124);
    buf.swap(33, 66);
    buf.swap(35, 98);
    buf.swap(37, 82);
    buf.swap(38, 50);
    buf.swap(39, 114);
    buf.swap(41, 74);
    buf.swap(43, 106);
    buf.swap(45, 90);
    buf.swap(46, 58);
    buf.swap(47, 122);
    buf.swap(49, 70);
    buf.swap(51, 102);
    buf.swap(53, 86);
    buf.swap(55, 118);
    buf.swap(57, 78);
    buf.swap(59, 110);
    buf.swap(61, 94);
    buf.swap(63, 126);
    buf.swap(67, 97);
    buf.swap(69, 81);
    buf.swap(71, 113);
    buf.swap(75, 105);
    buf.swap(77, 89);
    buf.swap(79, 121);
    buf.swap(83, 101);
    buf.swap(87, 117);
    buf.swap(91, 109);
    buf.swap(95, 125);
    buf.swap(103, 115);
    buf.swap(111, 123);
}

/// Fully unrolled bit reversal for size 256 (log_n = 8)
fn bit_rev_256<T>(buf: &mut [T]) {
    // Pre-computed bit-reversed pairs for N=256
    // Only swap when i < rev(i) to avoid double swapping
    buf.swap(1, 128);
    buf.swap(2, 64);
    buf.swap(3, 192);
    buf.swap(4, 32);
    buf.swap(5, 160);
    buf.swap(6, 96);
    buf.swap(7, 224);
    buf.swap(8, 16);
    buf.swap(9, 144);
    buf.swap(10, 80);
    buf.swap(11, 208);
    buf.swap(12, 48);
    buf.swap(13, 176);
    buf.swap(14, 112);
    buf.swap(15, 240);
    buf.swap(17, 136);
    buf.swap(18, 72);
    buf.swap(19, 200);
    buf.swap(20, 40);
    buf.swap(21, 168);
    buf.swap(22, 104);
    buf.swap(23, 232);
    buf.swap(25, 152);
    buf.swap(26, 88);
    buf.swap(27, 216);
    buf.swap(28, 56);
    buf.swap(29, 184);
    buf.swap(30, 120);
    buf.swap(31, 248);
    buf.swap(33, 132);
    buf.swap(34, 68);
    buf.swap(35, 196);
    buf.swap(37, 164);
    buf.swap(38, 100);
    buf.swap(39, 228);
    buf.swap(41, 148);
    buf.swap(42, 84);
    buf.swap(43, 212);
    buf.swap(44, 52);
    buf.swap(45, 180);
    buf.swap(46, 116);
    buf.swap(47, 244);
    buf.swap(49, 140);
    buf.swap(50, 76);
    buf.swap(51, 204);
    buf.swap(53, 172);
    buf.swap(54, 108);
    buf.swap(55, 236);
    buf.swap(57, 156);
    buf.swap(58, 92);
    buf.swap(59, 220);
    buf.swap(61, 188);
    buf.swap(62, 124);
    buf.swap(63, 252);
    buf.swap(65, 130);
    buf.swap(67, 194);
    buf.swap(69, 162);
    buf.swap(70, 98);
    buf.swap(71, 226);
    buf.swap(73, 146);
    buf.swap(74, 82);
    buf.swap(75, 210);
    buf.swap(77, 178);
    buf.swap(78, 114);
    buf.swap(79, 242);
    buf.swap(81, 138);
    buf.swap(83, 202);
    buf.swap(85, 170);
    buf.swap(86, 106);
    buf.swap(87, 234);
    buf.swap(89, 154);
    buf.swap(91, 218);
    buf.swap(93, 186);
    buf.swap(94, 122);
    buf.swap(95, 250);
    buf.swap(97, 134);
    buf.swap(99, 198);
    buf.swap(101, 166);
    buf.swap(103, 230);
    buf.swap(105, 150);
    buf.swap(107, 214);
    buf.swap(109, 182);
    buf.swap(110, 118);
    buf.swap(111, 246);
    buf.swap(113, 142);
    buf.swap(115, 206);
    buf.swap(117, 174);
    buf.swap(119, 238);
    buf.swap(121, 158);
    buf.swap(123, 222);
    buf.swap(125, 190);
    buf.swap(127, 254);
    buf.swap(131, 193);
    buf.swap(133, 161);
    buf.swap(135, 225);
    buf.swap(137, 145);
    buf.swap(139, 209);
    buf.swap(141, 177);
    buf.swap(143, 241);
    buf.swap(147, 201);
    buf.swap(149, 169);
    buf.swap(151, 233);
    buf.swap(155, 217);
    buf.swap(157, 185);
    buf.swap(159, 249);
    buf.swap(163, 197);
    buf.swap(167, 229);
    buf.swap(171, 213);
    buf.swap(173, 181);
    buf.swap(175, 245);
    buf.swap(179, 205);
    buf.swap(183, 237);
    buf.swap(187, 221);
    buf.swap(191, 253);
    buf.swap(199, 227);
    buf.swap(203, 211);
    buf.swap(207, 243);
    buf.swap(215, 235);
    buf.swap(223, 251);
    buf.swap(239, 247);
}

/// Fully unrolled bit reversal for size 512 (log_n = 9)
fn bit_rev_512<T>(buf: &mut [T]) {
    // Pre-computed bit-reversed pairs for N=512
    // Only swap when i < rev(i) to avoid double swapping
    buf.swap(1, 256);
    buf.swap(2, 128);
    buf.swap(3, 384);
    buf.swap(4, 64);
    buf.swap(5, 320);
    buf.swap(6, 192);
    buf.swap(7, 448);
    buf.swap(8, 32);
    buf.swap(9, 288);
    buf.swap(10, 160);
    buf.swap(11, 416);
    buf.swap(12, 96);
    buf.swap(13, 352);
    buf.swap(14, 224);
    buf.swap(15, 480);
    buf.swap(17, 272);
    buf.swap(18, 144);
    buf.swap(19, 400);
    buf.swap(20, 80);
    buf.swap(21, 336);
    buf.swap(22, 208);
    buf.swap(23, 464);
    buf.swap(24, 48);
    buf.swap(25, 304);
    buf.swap(26, 176);
    buf.swap(27, 432);
    buf.swap(28, 112);
    buf.swap(29, 368);
    buf.swap(30, 240);
    buf.swap(31, 496);
    buf.swap(33, 264);
    buf.swap(34, 136);
    buf.swap(35, 392);
    buf.swap(36, 72);
    buf.swap(37, 328);
    buf.swap(38, 200);
    buf.swap(39, 456);
    buf.swap(41, 296);
    buf.swap(42, 168);
    buf.swap(43, 424);
    buf.swap(44, 104);
    buf.swap(45, 360);
    buf.swap(46, 232);
    buf.swap(47, 488);
    buf.swap(49, 280);
    buf.swap(50, 152);
    buf.swap(51, 408);
    buf.swap(52, 88);
    buf.swap(53, 344);
    buf.swap(54, 216);
    buf.swap(55, 472);
    buf.swap(57, 312);
    buf.swap(58, 184);
    buf.swap(59, 440);
    buf.swap(60, 120);
    buf.swap(61, 376);
    buf.swap(62, 248);
    buf.swap(63, 504);
    buf.swap(65, 260);
    buf.swap(66, 132);
    buf.swap(67, 388);
    buf.swap(69, 324);
    buf.swap(70, 196);
    buf.swap(71, 452);
    buf.swap(73, 292);
    buf.swap(74, 164);
    buf.swap(75, 420);
    buf.swap(76, 100);
    buf.swap(77, 356);
    buf.swap(78, 228);
    buf.swap(79, 484);
    buf.swap(81, 276);
    buf.swap(82, 148);
    buf.swap(83, 404);
    buf.swap(85, 340);
    buf.swap(86, 212);
    buf.swap(87, 468);
    buf.swap(89, 308);
    buf.swap(90, 180);
    buf.swap(91, 436);
    buf.swap(92, 116);
    buf.swap(93, 372);
    buf.swap(94, 244);
    buf.swap(95, 500);
    buf.swap(97, 268);
    buf.swap(98, 140);
    buf.swap(99, 396);
    buf.swap(101, 332);
    buf.swap(102, 204);
    buf.swap(103, 460);
    buf.swap(105, 300);
    buf.swap(106, 172);
    buf.swap(107, 428);
    buf.swap(109, 364);
    buf.swap(110, 236);
    buf.swap(111, 492);
    buf.swap(113, 284);
    buf.swap(114, 156);
    buf.swap(115, 412);
    buf.swap(117, 348);
    buf.swap(118, 220);
    buf.swap(119, 476);
    buf.swap(121, 316);
    buf.swap(122, 188);
    buf.swap(123, 444);
    buf.swap(125, 380);
    buf.swap(126, 252);
    buf.swap(127, 508);
    buf.swap(129, 258);
    buf.swap(131, 386);
    buf.swap(133, 322);
    buf.swap(134, 194);
    buf.swap(135, 450);
    buf.swap(137, 290);
    buf.swap(138, 162);
    buf.swap(139, 418);
    buf.swap(141, 354);
    buf.swap(142, 226);
    buf.swap(143, 482);
    buf.swap(145, 274);
    buf.swap(147, 402);
    buf.swap(149, 338);
    buf.swap(150, 210);
    buf.swap(151, 466);
    buf.swap(153, 306);
    buf.swap(154, 178);
    buf.swap(155, 434);
    buf.swap(157, 370);
    buf.swap(158, 242);
    buf.swap(159, 498);
    buf.swap(161, 266);
    buf.swap(163, 394);
    buf.swap(165, 330);
    buf.swap(166, 202);
    buf.swap(167, 458);
    buf.swap(169, 298);
    buf.swap(171, 426);
    buf.swap(173, 362);
    buf.swap(174, 234);
    buf.swap(175, 490);
    buf.swap(177, 282);
    buf.swap(179, 410);
    buf.swap(181, 346);
    buf.swap(182, 218);
    buf.swap(183, 474);
    buf.swap(185, 314);
    buf.swap(187, 442);
    buf.swap(189, 378);
    buf.swap(190, 250);
    buf.swap(191, 506);
    buf.swap(193, 262);
    buf.swap(195, 390);
    buf.swap(197, 326);
    buf.swap(199, 454);
    buf.swap(201, 294);
    buf.swap(203, 422);
    buf.swap(205, 358);
    buf.swap(206, 230);
    buf.swap(207, 486);
    buf.swap(209, 278);
    buf.swap(211, 406);
    buf.swap(213, 342);
    buf.swap(215, 470);
    buf.swap(217, 310);
    buf.swap(219, 438);
    buf.swap(221, 374);
    buf.swap(222, 246);
    buf.swap(223, 502);
    buf.swap(225, 270);
    buf.swap(227, 398);
    buf.swap(229, 334);
    buf.swap(231, 462);
    buf.swap(233, 302);
    buf.swap(235, 430);
    buf.swap(237, 366);
    buf.swap(239, 494);
    buf.swap(241, 286);
    buf.swap(243, 414);
    buf.swap(245, 350);
    buf.swap(247, 478);
    buf.swap(249, 318);
    buf.swap(251, 446);
    buf.swap(253, 382);
    buf.swap(255, 510);
    buf.swap(259, 385);
    buf.swap(261, 321);
    buf.swap(263, 449);
    buf.swap(265, 289);
    buf.swap(267, 417);
    buf.swap(269, 353);
    buf.swap(271, 481);
    buf.swap(275, 401);
    buf.swap(277, 337);
    buf.swap(279, 465);
    buf.swap(281, 305);
    buf.swap(283, 433);
    buf.swap(285, 369);
    buf.swap(287, 497);
    buf.swap(291, 393);
    buf.swap(293, 329);
    buf.swap(295, 457);
    buf.swap(299, 425);
    buf.swap(301, 361);
    buf.swap(303, 489);
    buf.swap(307, 409);
    buf.swap(309, 345);
    buf.swap(311, 473);
    buf.swap(315, 441);
    buf.swap(317, 377);
    buf.swap(319, 505);
    buf.swap(323, 389);
    buf.swap(327, 453);
    buf.swap(331, 421);
    buf.swap(333, 357);
    buf.swap(335, 485);
    buf.swap(339, 405);
    buf.swap(343, 469);
    buf.swap(347, 437);
    buf.swap(349, 373);
    buf.swap(351, 501);
    buf.swap(355, 397);
    buf.swap(359, 461);
    buf.swap(363, 429);
    buf.swap(367, 493);
    buf.swap(371, 413);
    buf.swap(375, 477);
    buf.swap(379, 445);
    buf.swap(383, 509);
    buf.swap(391, 451);
    buf.swap(395, 419);
    buf.swap(399, 483);
    buf.swap(407, 467);
    buf.swap(411, 435);
    buf.swap(415, 499);
    buf.swap(423, 459);
    buf.swap(431, 491);
    buf.swap(439, 475);
    buf.swap(447, 507);
    buf.swap(463, 487);
    buf.swap(479, 503);
}

/// Fully unrolled bit reversal for size 1024 (log_n = 10)
fn bit_rev_1024<T>(buf: &mut [T]) {
    // Pre-computed bit-reversed pairs for N=1024
    // Only swap when i < rev(i) to avoid double swapping
    buf.swap(1, 512);
    buf.swap(2, 256);
    buf.swap(3, 768);
    buf.swap(4, 128);
    buf.swap(5, 640);
    buf.swap(6, 384);
    buf.swap(7, 896);
    buf.swap(8, 64);
    buf.swap(9, 576);
    buf.swap(10, 320);
    buf.swap(11, 832);
    buf.swap(12, 192);
    buf.swap(13, 704);
    buf.swap(14, 448);
    buf.swap(15, 960);
    buf.swap(16, 32);
    buf.swap(17, 544);
    buf.swap(18, 288);
    buf.swap(19, 800);
    buf.swap(20, 160);
    buf.swap(21, 672);
    buf.swap(22, 416);
    buf.swap(23, 928);
    buf.swap(24, 96);
    buf.swap(25, 608);
    buf.swap(26, 352);
    buf.swap(27, 864);
    buf.swap(28, 224);
    buf.swap(29, 736);
    buf.swap(30, 480);
    buf.swap(31, 992);
    buf.swap(33, 528);
    buf.swap(34, 272);
    buf.swap(35, 784);
    buf.swap(36, 144);
    buf.swap(37, 656);
    buf.swap(38, 400);
    buf.swap(39, 912);
    buf.swap(40, 80);
    buf.swap(41, 592);
    buf.swap(42, 336);
    buf.swap(43, 848);
    buf.swap(44, 208);
    buf.swap(45, 720);
    buf.swap(46, 464);
    buf.swap(47, 976);
    buf.swap(49, 560);
    buf.swap(50, 304);
    buf.swap(51, 816);
    buf.swap(52, 176);
    buf.swap(53, 688);
    buf.swap(54, 432);
    buf.swap(55, 944);
    buf.swap(56, 112);
    buf.swap(57, 624);
    buf.swap(58, 368);
    buf.swap(59, 880);
    buf.swap(60, 240);
    buf.swap(61, 752);
    buf.swap(62, 496);
    buf.swap(63, 1008);
    buf.swap(65, 520);
    buf.swap(66, 264);
    buf.swap(67, 776);
    buf.swap(68, 136);
    buf.swap(69, 648);
    buf.swap(70, 392);
    buf.swap(71, 904);
    buf.swap(73, 584);
    buf.swap(74, 328);
    buf.swap(75, 840);
    buf.swap(76, 200);
    buf.swap(77, 712);
    buf.swap(78, 456);
    buf.swap(79, 968);
    buf.swap(81, 552);
    buf.swap(82, 296);
    buf.swap(83, 808);
    buf.swap(84, 168);
    buf.swap(85, 680);
    buf.swap(86, 424);
    buf.swap(87, 936);
    buf.swap(88, 104);
    buf.swap(89, 616);
    buf.swap(90, 360);
    buf.swap(91, 872);
    buf.swap(92, 232);
    buf.swap(93, 744);
    buf.swap(94, 488);
    buf.swap(95, 1000);
    buf.swap(97, 536);
    buf.swap(98, 280);
    buf.swap(99, 792);
    buf.swap(100, 152);
    buf.swap(101, 664);
    buf.swap(102, 408);
    buf.swap(103, 920);
    buf.swap(105, 600);
    buf.swap(106, 344);
    buf.swap(107, 856);
    buf.swap(108, 216);
    buf.swap(109, 728);
    buf.swap(110, 472);
    buf.swap(111, 984);
    buf.swap(113, 568);
    buf.swap(114, 312);
    buf.swap(115, 824);
    buf.swap(116, 184);
    buf.swap(117, 696);
    buf.swap(118, 440);
    buf.swap(119, 952);
    buf.swap(121, 632);
    buf.swap(122, 376);
    buf.swap(123, 888);
    buf.swap(124, 248);
    buf.swap(125, 760);
    buf.swap(126, 504);
    buf.swap(127, 1016);
    buf.swap(129, 516);
    buf.swap(130, 260);
    buf.swap(131, 772);
    buf.swap(133, 644);
    buf.swap(134, 388);
    buf.swap(135, 900);
    buf.swap(137, 580);
    buf.swap(138, 324);
    buf.swap(139, 836);
    buf.swap(140, 196);
    buf.swap(141, 708);
    buf.swap(142, 452);
    buf.swap(143, 964);
    buf.swap(145, 548);
    buf.swap(146, 292);
    buf.swap(147, 804);
    buf.swap(148, 164);
    buf.swap(149, 676);
    buf.swap(150, 420);
    buf.swap(151, 932);
    buf.swap(153, 612);
    buf.swap(154, 356);
    buf.swap(155, 868);
    buf.swap(156, 228);
    buf.swap(157, 740);
    buf.swap(158, 484);
    buf.swap(159, 996);
    buf.swap(161, 532);
    buf.swap(162, 276);
    buf.swap(163, 788);
    buf.swap(165, 660);
    buf.swap(166, 404);
    buf.swap(167, 916);
    buf.swap(169, 596);
    buf.swap(170, 340);
    buf.swap(171, 852);
    buf.swap(172, 212);
    buf.swap(173, 724);
    buf.swap(174, 468);
    buf.swap(175, 980);
    buf.swap(177, 564);
    buf.swap(178, 308);
    buf.swap(179, 820);
    buf.swap(181, 692);
    buf.swap(182, 436);
    buf.swap(183, 948);
    buf.swap(185, 628);
    buf.swap(186, 372);
    buf.swap(187, 884);
    buf.swap(188, 244);
    buf.swap(189, 756);
    buf.swap(190, 500);
    buf.swap(191, 1012);
    buf.swap(193, 524);
    buf.swap(194, 268);
    buf.swap(195, 780);
    buf.swap(197, 652);
    buf.swap(198, 396);
    buf.swap(199, 908);
    buf.swap(201, 588);
    buf.swap(202, 332);
    buf.swap(203, 844);
    buf.swap(205, 716);
    buf.swap(206, 460);
    buf.swap(207, 972);
    buf.swap(209, 556);
    buf.swap(210, 300);
    buf.swap(211, 812);
    buf.swap(213, 684);
    buf.swap(214, 428);
    buf.swap(215, 940);
    buf.swap(217, 620);
    buf.swap(218, 364);
    buf.swap(219, 876);
    buf.swap(220, 236);
    buf.swap(221, 748);
    buf.swap(222, 492);
    buf.swap(223, 1004);
    buf.swap(225, 540);
    buf.swap(226, 284);
    buf.swap(227, 796);
    buf.swap(229, 668);
    buf.swap(230, 412);
    buf.swap(231, 924);
    buf.swap(233, 604);
    buf.swap(234, 348);
    buf.swap(235, 860);
    buf.swap(237, 732);
    buf.swap(238, 476);
    buf.swap(239, 988);
    buf.swap(241, 572);
    buf.swap(242, 316);
    buf.swap(243, 828);
    buf.swap(245, 700);
    buf.swap(246, 444);
    buf.swap(247, 956);
    buf.swap(249, 636);
    buf.swap(250, 380);
    buf.swap(251, 892);
    buf.swap(253, 764);
    buf.swap(254, 508);
    buf.swap(255, 1020);
    buf.swap(257, 514);
    buf.swap(259, 770);
    buf.swap(261, 642);
    buf.swap(262, 386);
    buf.swap(263, 898);
    buf.swap(265, 578);
    buf.swap(266, 322);
    buf.swap(267, 834);
    buf.swap(269, 706);
    buf.swap(270, 450);
    buf.swap(271, 962);
    buf.swap(273, 546);
    buf.swap(274, 290);
    buf.swap(275, 802);
    buf.swap(277, 674);
    buf.swap(278, 418);
    buf.swap(279, 930);
    buf.swap(281, 610);
    buf.swap(282, 354);
    buf.swap(283, 866);
    buf.swap(285, 738);
    buf.swap(286, 482);
    buf.swap(287, 994);
    buf.swap(289, 530);
    buf.swap(291, 786);
    buf.swap(293, 658);
    buf.swap(294, 402);
    buf.swap(295, 914);
    buf.swap(297, 594);
    buf.swap(298, 338);
    buf.swap(299, 850);
    buf.swap(301, 722);
    buf.swap(302, 466);
    buf.swap(303, 978);
    buf.swap(305, 562);
    buf.swap(307, 818);
    buf.swap(309, 690);
    buf.swap(310, 434);
    buf.swap(311, 946);
    buf.swap(313, 626);
    buf.swap(314, 370);
    buf.swap(315, 882);
    buf.swap(317, 754);
    buf.swap(318, 498);
    buf.swap(319, 1010);
    buf.swap(321, 522);
    buf.swap(323, 778);
    buf.swap(325, 650);
    buf.swap(326, 394);
    buf.swap(327, 906);
    buf.swap(329, 586);
    buf.swap(331, 842);
    buf.swap(333, 714);
    buf.swap(334, 458);
    buf.swap(335, 970);
    buf.swap(337, 554);
    buf.swap(339, 810);
    buf.swap(341, 682);
    buf.swap(342, 426);
    buf.swap(343, 938);
    buf.swap(345, 618);
    buf.swap(346, 362);
    buf.swap(347, 874);
    buf.swap(349, 746);
    buf.swap(350, 490);
    buf.swap(351, 1002);
    buf.swap(353, 538);
    buf.swap(355, 794);
    buf.swap(357, 666);
    buf.swap(358, 410);
    buf.swap(359, 922);
    buf.swap(361, 602);
    buf.swap(363, 858);
    buf.swap(365, 730);
    buf.swap(366, 474);
    buf.swap(367, 986);
    buf.swap(369, 570);
    buf.swap(371, 826);
    buf.swap(373, 698);
    buf.swap(374, 442);
    buf.swap(375, 954);
    buf.swap(377, 634);
    buf.swap(379, 890);
    buf.swap(381, 762);
    buf.swap(382, 506);
    buf.swap(383, 1018);
    buf.swap(385, 518);
    buf.swap(387, 774);
    buf.swap(389, 646);
    buf.swap(391, 902);
    buf.swap(393, 582);
    buf.swap(395, 838);
    buf.swap(397, 710);
    buf.swap(398, 454);
    buf.swap(399, 966);
    buf.swap(401, 550);
    buf.swap(403, 806);
    buf.swap(405, 678);
    buf.swap(406, 422);
    buf.swap(407, 934);
    buf.swap(409, 614);
    buf.swap(411, 870);
    buf.swap(413, 742);
    buf.swap(414, 486);
    buf.swap(415, 998);
    buf.swap(417, 534);
    buf.swap(419, 790);
    buf.swap(421, 662);
    buf.swap(423, 918);
    buf.swap(425, 598);
    buf.swap(427, 854);
    buf.swap(429, 726);
    buf.swap(430, 470);
    buf.swap(431, 982);
    buf.swap(433, 566);
    buf.swap(435, 822);
    buf.swap(437, 694);
    buf.swap(439, 950);
    buf.swap(441, 630);
    buf.swap(443, 886);
    buf.swap(445, 758);
    buf.swap(446, 502);
    buf.swap(447, 1014);
    buf.swap(449, 526);
    buf.swap(451, 782);
    buf.swap(453, 654);
    buf.swap(455, 910);
    buf.swap(457, 590);
    buf.swap(459, 846);
    buf.swap(461, 718);
    buf.swap(463, 974);
    buf.swap(465, 558);
    buf.swap(467, 814);
    buf.swap(469, 686);
    buf.swap(471, 942);
    buf.swap(473, 622);
    buf.swap(475, 878);
    buf.swap(477, 750);
    buf.swap(478, 494);
    buf.swap(479, 1006);
    buf.swap(481, 542);
    buf.swap(483, 798);
    buf.swap(485, 670);
    buf.swap(487, 926);
    buf.swap(489, 606);
    buf.swap(491, 862);
    buf.swap(493, 734);
    buf.swap(495, 990);
    buf.swap(497, 574);
    buf.swap(499, 830);
    buf.swap(501, 702);
    buf.swap(503, 958);
    buf.swap(505, 638);
    buf.swap(507, 894);
    buf.swap(509, 766);
    buf.swap(511, 1022);
    buf.swap(515, 769);
    buf.swap(517, 641);
    buf.swap(519, 897);
    buf.swap(521, 577);
    buf.swap(523, 833);
    buf.swap(525, 705);
    buf.swap(527, 961);
    buf.swap(529, 545);
    buf.swap(531, 801);
    buf.swap(533, 673);
    buf.swap(535, 929);
    buf.swap(537, 609);
    buf.swap(539, 865);
    buf.swap(541, 737);
    buf.swap(543, 993);
    buf.swap(547, 785);
    buf.swap(549, 657);
    buf.swap(551, 913);
    buf.swap(553, 593);
    buf.swap(555, 849);
    buf.swap(557, 721);
    buf.swap(559, 977);
    buf.swap(563, 817);
    buf.swap(565, 689);
    buf.swap(567, 945);
    buf.swap(569, 625);
    buf.swap(571, 881);
    buf.swap(573, 753);
    buf.swap(575, 1009);
    buf.swap(579, 777);
    buf.swap(581, 649);
    buf.swap(583, 905);
    buf.swap(587, 841);
    buf.swap(589, 713);
    buf.swap(591, 969);
    buf.swap(595, 809);
    buf.swap(597, 681);
    buf.swap(599, 937);
    buf.swap(601, 617);
    buf.swap(603, 873);
    buf.swap(605, 745);
    buf.swap(607, 1001);
    buf.swap(611, 793);
    buf.swap(613, 665);
    buf.swap(615, 921);
    buf.swap(619, 857);
    buf.swap(621, 729);
    buf.swap(623, 985);
    buf.swap(627, 825);
    buf.swap(629, 697);
    buf.swap(631, 953);
    buf.swap(635, 889);
    buf.swap(637, 761);
    buf.swap(639, 1017);
    buf.swap(643, 773);
    buf.swap(647, 901);
    buf.swap(651, 837);
    buf.swap(653, 709);
    buf.swap(655, 965);
    buf.swap(659, 805);
    buf.swap(661, 677);
    buf.swap(663, 933);
    buf.swap(667, 869);
    buf.swap(669, 741);
    buf.swap(671, 997);
    buf.swap(675, 789);
    buf.swap(679, 917);
    buf.swap(683, 853);
    buf.swap(685, 725);
    buf.swap(687, 981);
    buf.swap(691, 821);
    buf.swap(695, 949);
    buf.swap(699, 885);
    buf.swap(701, 757);
    buf.swap(703, 1013);
    buf.swap(707, 781);
    buf.swap(711, 909);
    buf.swap(715, 845);
    buf.swap(719, 973);
    buf.swap(723, 813);
    buf.swap(727, 941);
    buf.swap(731, 877);
    buf.swap(733, 749);
    buf.swap(735, 1005);
    buf.swap(739, 797);
    buf.swap(743, 925);
    buf.swap(747, 861);
    buf.swap(751, 989);
    buf.swap(755, 829);
    buf.swap(759, 957);
    buf.swap(763, 893);
    buf.swap(767, 1021);
    buf.swap(775, 899);
    buf.swap(779, 835);
    buf.swap(783, 963);
    buf.swap(787, 803);
    buf.swap(791, 931);
    buf.swap(795, 867);
    buf.swap(799, 995);
    buf.swap(807, 915);
    buf.swap(811, 851);
    buf.swap(815, 979);
    buf.swap(823, 947);
    buf.swap(827, 883);
    buf.swap(831, 1011);
    buf.swap(839, 907);
    buf.swap(847, 971);
    buf.swap(855, 939);
    buf.swap(859, 875);
    buf.swap(863, 1003);
    buf.swap(871, 923);
    buf.swap(879, 987);
    buf.swap(887, 955);
    buf.swap(895, 1019);
    buf.swap(911, 967);
    buf.swap(919, 935);
    buf.swap(927, 999);
    buf.swap(943, 983);
    buf.swap(959, 1015);
    buf.swap(991, 1007);
}

/// Dispatches to fully unrolled versions for small sizes
pub(crate) fn bit_rev_unrolled<T>(buf: &mut [T], log_n: usize) {
    match log_n {
        6 => bit_rev_64(buf),
        7 => bit_rev_128(buf),
        8 => bit_rev_256(buf),
        9 => bit_rev_512(buf),
        10 => bit_rev_1024(buf),
        _ => bit_rev_gray(buf, log_n),
    }
}

/// ## References
/// [1] <https://www.katjaas.nl/bitreversal/bitreversal.html>
pub(crate) fn bit_rev_gray<T>(buf: &mut [T], log_n: usize) {
    let mut nodd: usize;
    let mut noddrev; // to hold bitwise negated or odd values

    let big_n = 1 << log_n;
    let halfn = big_n >> 1; // frequently used 'constants'
    let quartn = big_n >> 2;
    let nmin1 = big_n - 1;

    let mut forward = halfn; // variable initialisations
    let mut rev = 1;

    let mut i: usize = quartn;
    while i > 0 {
        // start of bit reversed permutation loop, N/4 iterations
        // Gray code generator for even values:

        let zeros = i.trailing_zeros();
        forward ^= 2 << zeros; // toggle one bit of forward
        rev ^= quartn >> zeros; // toggle one bit of rev

        // swap even and ~even conditionally
        if forward < rev {
            buf.swap(forward, rev);
            nodd = nmin1 ^ forward; // compute the bitwise negations
            noddrev = nmin1 ^ rev;
            buf.swap(nodd, noddrev); // swap bitwise-negated pairs
        }

        nodd = forward ^ 1; // compute the odd values from the even
        noddrev = rev ^ halfn;

        // swap odd unconditionally
        buf.swap(nodd, noddrev);
        i -= 1;
    }
}

/// Slow, naive implementation of bit reversal
pub(crate) fn bit_rev_naive<T>(buf: &mut [T], _log_n: usize) {
    let n = buf.len();
    let mut j = 0;

    for i in 1..n {
        let mut bit = n >> 1;

        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if i < j {
            buf.swap(i, j);
        }
    }
}

/// Pure Rust implementation of Cache Optimal Bit-Reverse Algorithm (COBRA).
/// Rewritten from a C++ implementation [3].
///
/// ## References
/// [1] L. Carter and K. S. Gatlin, "Towards an optimal bit-reversal permutation program," Proceedings 39th Annual
/// Symposium on Foundations of Computer Science (Cat. No.98CB36280), Palo Alto, CA, USA, 1998, pp. 544-553, doi:
/// 10.1109/SFCS.1998.743505.
/// [2] Christian Knauth, Boran Adas, Daniel Whitfield, Xuesong Wang, Lydia Ickler, Tim Conrad, Oliver Serang:
/// Practically efficient methods for performing bit-reversed permutation in C++11 on the x86-64 architecture
/// [3] <https://bitbucket.org/orserang/bit-reversal-methods/src/master/src_and_bin/src/algorithms/COBRAShuffle.hpp>
#[multiversion::multiversion(targets("x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
                                     "x86_64+avx2+fma", // x86_64-v3
                                     "x86_64+sse4.2", // x86_64-v2
                                     "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+gfni",
                                     "x86+avx2+fma",
                                     "x86+sse4.2",
                                     "x86+sse2",
))]
pub fn bit_rev_cobra<T: Default + Copy + Clone>(v: &mut [T], log_n: usize) {
    assert!(BLOCK_WIDTH == 1 << LOG_BLOCK_WIDTH);
    if log_n <= 2 * LOG_BLOCK_WIDTH {
        bit_rev_gray(v, log_n);
        return;
    }
    let num_b_bits = log_n - 2 * LOG_BLOCK_WIDTH;
    let b_size: usize = 1 << num_b_bits;

    let mut buffer = [T::default(); BLOCK_WIDTH * BLOCK_WIDTH];

    for b in 0..b_size {
        let b_rev = b.reverse_bits() >> ((b_size - 1).leading_zeros());

        // Copy block to buffer
        for a in 0..BLOCK_WIDTH {
            let a_rev = a.reverse_bits() >> ((BLOCK_WIDTH - 1).leading_zeros());
            for c in 0..BLOCK_WIDTH {
                buffer[(a_rev << LOG_BLOCK_WIDTH) | c] =
                    v[(a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c];
            }
        }

        for c in 0..BLOCK_WIDTH {
            // NOTE: Typo in original pseudocode by Carter and Gatlin at the following line:
            let c_rev = c.reverse_bits() >> ((BLOCK_WIDTH - 1).leading_zeros());

            for a_rev in 0..BLOCK_WIDTH {
                let a = a_rev.reverse_bits() >> ((BLOCK_WIDTH - 1).leading_zeros());

                // To guarantee each value is swapped only one time:
                // index < reversed_index <-->
                // a b c < c' b' a' <-->
                // a < c' ||
                // a <= c' && b < b' ||
                // a <= c' && b <= b' && a' < c
                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    let v_idx = (c_rev << num_b_bits << LOG_BLOCK_WIDTH)
                        | (b_rev << LOG_BLOCK_WIDTH)
                        | a_rev;
                    let b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;
                    std::mem::swap(&mut v[v_idx], &mut buffer[b_idx]);
                }
            }
        }

        // Copy changes that were swapped into buffer above:
        for a in 0..BLOCK_WIDTH {
            let a_rev = a.reverse_bits() >> ((BLOCK_WIDTH - 1).leading_zeros());
            for c in 0..BLOCK_WIDTH {
                let c_rev = c.reverse_bits() >> ((BLOCK_WIDTH - 1).leading_zeros());
                let index_less_than_reverse = a < c_rev
                    || (a == c_rev && b < b_rev)
                    || (a == c_rev && b == b_rev && a_rev < c);

                if index_less_than_reverse {
                    let v_idx = (a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c;
                    let b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;
                    std::mem::swap(&mut v[v_idx], &mut buffer[b_idx]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Top down bit reverse interleaving. This is a very simple and well known approach that we
    /// only use for testing COBRA and any other bit reverse algorithms.
    fn top_down_bit_reverse_permutation<T: Copy + Clone>(x: &[T]) -> Vec<T> {
        if x.len() == 1 {
            return x.to_vec();
        }

        let mut y = Vec::with_capacity(x.len());
        let mut evens = Vec::with_capacity(x.len() >> 1);
        let mut odds = Vec::with_capacity(x.len() >> 1);

        let mut i = 1;
        while i < x.len() {
            evens.push(x[i - 1]);
            odds.push(x[i]);
            i += 2;
        }

        y.extend_from_slice(&top_down_bit_reverse_permutation(&evens));
        y.extend_from_slice(&top_down_bit_reverse_permutation(&odds));
        y
    }

    #[test]
    fn test_cobra_bit_reversal() {
        for n in 4..23 {
            let big_n = 1 << n;
            let mut v: Vec<_> = (0..big_n).collect();
            bit_rev_cobra(&mut v, n);

            let x: Vec<_> = (0..big_n).collect();
            assert_eq!(v, top_down_bit_reverse_permutation(&x));
        }
    }

    #[test]
    fn test_gray_bit_reversal() {
        let n = 3;
        let big_n = 1 << n;
        let mut buf: Vec<f64> = (0..big_n).map(f64::from).collect();
        bit_rev_gray(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(buf, vec![0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]);

        let n = 4;
        let big_n = 1 << n;
        let mut buf: Vec<f64> = (0..big_n).map(f64::from).collect();
        bit_rev_gray(&mut buf, n);
        println!("{buf:?}");
        assert_eq!(
            buf,
            vec![
                0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0,
                15.0,
            ]
        );
    }

    #[test]
    fn test_naive_bit_reversal() {
        for n in 2..24 {
            let big_n = 1 << n;
            let mut actual_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let mut actual_im: Vec<f64> = (0..big_n).map(f64::from).collect();

            bit_rev_naive(&mut actual_re, n);

            bit_rev_naive(&mut actual_im, n);

            let input_re: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_re = top_down_bit_reverse_permutation(&input_re);
            assert_eq!(actual_re, expected_re);

            let input_im: Vec<f64> = (0..big_n).map(f64::from).collect();
            let expected_im = top_down_bit_reverse_permutation(&input_im);
            assert_eq!(actual_im, expected_im);
        }
    }
}
