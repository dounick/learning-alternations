import json

alternating = "feed, give, lease, lend, loan, pass, pay, peddle, refund, render, rent, repay, sell, serve, trade, advance, allocate, allot, assign, award, bequeath, cede, concede, extend, grant, guarantee, issue, leave, offer, owe, promise, vote, will, yield, bring, take, forward, hand, mail, post, send, ship, slip, smuggle, sneak, bounce, float, roll, slide, carry, drag, haul, heave, heft, hoist, kick, lug, pull, push, schlep, shove, tote, tow, tug, barge, bus, cart, drive, ferry, fly, row, shuttle, truck, wheel, wire, bash, bat, bunt, catapult, chuck, flick, fling, flip, hit, hurl, kick, lob, pass, pitch, punt, shoot, shove, slam, slap, sling, throw, tip, toss, ask, cite, pose, preach, quote, read, relay, show, teach, tell, write, cable, e-mail, email, fax, modem, netmail, phone, radio, relay, satellite, semaphore, sign, signal, telephone, telecast, telegraph, telex, wire, wireless"
po_only = "address, administer, broadcast, convey, contribute, delegate, deliver, denounce, demonstrate, describe, dictate, dispatch, display, distribute, donate, elucidate, exhibit, express, explain, explicate, forfeit, illustrate, introduce, narrate, portray, proffer, recite, recommend, refer, reimburse, remit, restore, return, sacrifice, submit, surrender, transfer, transport, admit, allege, announce, articulate, assert, communicate, confess, convey, declare, mention, propose, recount, repeat, report, reveal, say, state, babble, bark, bawl, bellow, bleat, boom, bray, burble, cackle, call, carol, chant, chatter, chirp, cluck, coo, croak, croon, crow, cry, drawl, drone, gabble, gibber, groan, growl, grumble, grunt, hiss, holler, hoot, howl, jabber, lilt, lisp, moan, mumble, murmur, mutter, purr, rage, rasp, roar, rumble, scream, screech, shout, shriek, sing, snap, snarl, snuffle, splutter, squall, squawk, squeak, squeal, stammer, stutter, thunder, tisk, trill, trumpet, twitter, wail, warble, wheeze, whimper, whine, whisper, whistle, whoop, yammer, yap, yell, yelp, yodel, drop, hoist, lift, lower, raise, credit, entrust, furnish, issue, leave, present, provide, serve, supply, trust"
do_only = "accord, ask, bear, begrudge, bode, cost, deny, envy, flash, forbid, forgive, guarantee, issue, refuse, save, spare, vouchsafe, wish, write, bet, bill, charge, fine, mulct, overcharge, save, spare, tax, tip, undercharge, wager, acknowledge, adopt, appoint, consider, crown, deem, designate, elect, esteem, imagine, mark, nominate, ordain, proclaim, rate, reckon, report, want, anoint, baptize, brand, call, christen, consecrate, crown, decree, dub, label, make, name, nickname, pronounce, rule, stamp, style, term, vote, adjudge, ad judicate, assume, avow, believe, confess, declare, fancy, find, judge, presume, profess, prove, suppose, think, warrant"
benefactive_alternating = "arrange, assemble, bake, blow, build, carve, cast, chisel, churn, compile, cook, crochet, cut, develop, embroider, fashion, fold, forge, grind, grow, hack, hammer, hatch, knit, make, mold, pound, roll, sculpt, sew, shape, stitch, weave, whittle, design, dig, mint, bake, blend, boil, brew, clean, clear, cook, fix, fry, grill, hardboil, iron, light, mix, poach, pour, prepare, roast, roll, run, scramble, set, softboil, toast, wash, dance, draw, hum, paint, play, recite, sing, whistle, book, buy, call, cash, catch, charter, earn, fetch, find, gain, gather, get, hire, keep, lease, leave, order, procure, reach, rent, reserve, save, secure, slaughter, steal, vote, win"
benefactive_po_only = "accept, accumulate, acquire, appropriate, borrow, cadge, collect, exact, grab, inherit, obtain, purchase, receive, recover, regain, retrieve, seize, select, snatch, choose, designate, favor, indicate, prefer, coin, compose, compute, construct, create, derive, fabricate, form, invent, manufacture, mint, organize, produce, recreate, style, abduct, cadge, capture, confiscate, cop, emancipate, embezzle, exorcise, extort, extract, filch, flog, grab, impound, kidnap, liberate, lift, nab, pilfer, pinch, pirate, plagiarize, purloin, recover, redeem, reclaim, regain, repossess, rescue, retrieve, rustle, seize, smuggle, snatch, sneak, sponge, steal, swipe, take, thieve, wangle, weasel, winkle, withdraw, wrest"

alternating = alternating.split(", ")
po_only = po_only.split(", ")
do_only = do_only.split(", ")
benefactive_alternating = benefactive_alternating.split(", ")
benefactive_po_only = benefactive_po_only.split(", ")

po_only = [verb for verb in po_only if verb not in alternating]
do_only = [verb for verb in do_only if verb not in alternating]
benefactive_po_only = [verb for verb in benefactive_po_only if verb not in benefactive_alternating]

data = {
    "alternating": alternating,
    "po_only": po_only,
    "do_only": do_only,
    "benefactive_alternating": benefactive_alternating,
    "benefactive_po_only": benefactive_po_only
}

with open('./data/dative_verbs.json', 'w') as f:
    json.dump(data, f, indent=4)
